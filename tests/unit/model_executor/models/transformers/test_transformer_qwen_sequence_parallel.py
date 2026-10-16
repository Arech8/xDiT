"""Qwen-Image under sequence parallelism matches one rank, for prompts of any length, Edit-2511's zero_cond_t included.

Each rank's [text, image] chunk is split unevenly (TwoStreamSPSplit), so that the only SP pad trails the gathered
keys, where USP drops it. Here the ranks are lockstep threads with a model each, against a stand-in SP group whose
attention gathers every rank's K/V in rank order, as Ulysses and ring attention do, and honors `valid_kv_len` as USP
does. The real collectives run in the accelerator integration test.
"""

import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from xfuser.model_executor.models.transformers import transformer_qwen, transformers_utils
from xfuser.model_executor.models.transformers.transformer_qwen import xFuserQwenImageTransformerWrapper

_TINY_CONFIG = dict(
    patch_size=2,
    in_channels=8,
    out_channels=2,
    num_layers=2,
    attention_head_dim=16,
    num_attention_heads=2,
    joint_attention_dim=12,
    axes_dims_rope=(4, 6, 6),
)
# 64 image tokens either way, enough for a pad that fits on the last of 8 ranks: one image, or for zero_cond_t
# (Edit-2511) the edited image followed by a condition image, which the blocks modulate differently.
_IMG_SHAPES = {False: [(1, 8, 8)], True: [(1, 6, 8), (1, 4, 4)]}
_BATCH = 2
# Largest relative L2 error against one rank. fp32 throughout: measured <= 1.1e-7 with the pad dropped, ~1.1e-3 with
# it attended.
_TOLERANCE = 1e-5


class _SPRanks:
    """Lockstep threads standing in for the ranks of an SP group."""

    def __init__(self, size):
        self.size = size
        self._local = threading.local()
        self._barrier = threading.Barrier(size, timeout=120)
        self._slots = [None] * size

    @property
    def rank(self):
        return self._local.rank

    def exchange(self, x):
        """Every rank's `x`, in rank order."""
        self._slots[self.rank] = x
        self._barrier.wait()
        gathered = list(self._slots)
        # No rank may overwrite its slot before every rank has read it.
        self._barrier.wait()
        return gathered

    def run(self, fn):
        results, errors = [None] * self.size, []

        def worker(rank):
            self._local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as error:
                errors.append(error)
                self._barrier.abort()

        threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(self.size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise next((e for e in errors if not isinstance(e, threading.BrokenBarrierError)), errors[0])
        return results


def _install_sp(monkeypatch, ranks, trim_pad=True):
    def usp(query, key, value, dropout_p=0.0, is_causal=False, attn_layer=None, attention_kwargs=None):
        key = torch.cat(ranks.exchange(key), dim=2)
        value = torch.cat(ranks.exchange(value), dim=2)
        valid_kv_len = (attention_kwargs or {}).get("valid_kv_len")
        if trim_pad and valid_kv_len is not None:
            key, value = key[:, :, :valid_kv_len], value[:, :, :valid_kv_len]
        return F.scaled_dot_product_attention(query, key, value)

    monkeypatch.setattr(transformer_qwen, "USP", usp)
    # The diffusers norm and RoPE path, which the fused kernel replays on GPU.
    monkeypatch.setattr(transformer_qwen, "_HAS_FLYDSL", False)
    monkeypatch.setattr(transformer_qwen, "get_sequence_parallel_rank", lambda: ranks.rank)
    monkeypatch.setattr(transformer_qwen, "get_sequence_parallel_world_size", lambda: ranks.size)
    monkeypatch.setattr(
        transformers_utils,
        "get_sp_group",
        lambda: SimpleNamespace(all_gather=lambda x, dim: torch.cat(ranks.exchange(x), dim)),
    )


def _inputs(zero_cond_t, text_len):
    generator = torch.Generator().manual_seed(1)
    shapes = _IMG_SHAPES[zero_cond_t]
    image_len = sum(f * h * w for f, h, w in shapes)
    return dict(
        hidden_states=torch.randn(_BATCH, image_len, _TINY_CONFIG["in_channels"], generator=generator),
        encoder_hidden_states=torch.randn(_BATCH, text_len, _TINY_CONFIG["joint_attention_dim"], generator=generator),
        timestep=torch.rand(_BATCH, generator=generator),
        img_shapes=[shapes] * _BATCH,
        return_dict=False,
    )


@pytest.fixture(scope="module", params=[False, True], ids=["qwen-image", "zero-cond-t"])
def reference(request):
    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(**_TINY_CONFIG, zero_cond_t=request.param).eval()
    return SimpleNamespace(model=model, zero_cond_t=request.param)


def _refuse_empty_input(module, args):
    # As quantized GEMMs do: torchao's dynamic FP8 raises on a rank holding no text rows.
    assert args[0].numel(), "a Linear got an empty input"


def _run(monkeypatch, reference, inputs, sp_size, trim_pad=True):
    """Each rank's output. Every block's output is checked for the pad rows the test expects of its rank, and every
    Linear for an empty input."""
    ranks = _SPRanks(sp_size)
    _install_sp(monkeypatch, ranks, trim_pad)
    real_len = inputs["encoder_hidden_states"].shape[1] + inputs["hidden_states"].shape[1]
    pad = -real_len % sp_size

    def check_pad_rows(module, args, output):
        if ranks.rank == sp_size - 1 and pad:
            assert not output[1][:, -pad:].any(), "pad rows of the image stream kept"
            assert output[1][:, :-pad].any()

    models = []
    for _ in range(sp_size):
        model = xFuserQwenImageTransformerWrapper(**_TINY_CONFIG, zero_cond_t=reference.zero_cond_t).eval()
        model.load_state_dict(reference.model.state_dict())
        for block in model.transformer_blocks:
            # After the model's own hook, which zeroes the pad rows.
            block.register_forward_hook(check_pad_rows)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_pre_hook(_refuse_empty_input)
        models.append(model)

    @torch.no_grad()
    def forward(rank):
        return models[rank](**inputs)[0]

    return ranks.run(forward)


def _relative_error(actual, expected):
    return ((actual - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("text_len", [1, 30])
def test_one_rank_matches_diffusers(monkeypatch, reference, text_len):
    inputs = _inputs(reference.zero_cond_t, text_len)
    (output,) = _run(monkeypatch, reference, inputs, sp_size=1)
    with torch.no_grad():
        (expected,) = reference.model(**inputs)
    assert _relative_error(output, expected) < _TOLERANCE


@pytest.mark.parametrize("sp_size", [2, 3, 8])
# Shorter prompts than the SP degree leave ranks without text rows, whose text stream must skip its GEMMs; 30 and 32
# split into uneven and even shares.
@pytest.mark.parametrize("text_len", [1, 3, 7, 30, 32])
def test_every_rank_matches_one_rank(monkeypatch, reference, sp_size, text_len):
    inputs = _inputs(reference.zero_cond_t, text_len)
    (expected,) = _run(monkeypatch, reference, inputs, sp_size=1)
    for rank, output in enumerate(_run(monkeypatch, reference, inputs, sp_size)):
        assert output.shape == expected.shape
        assert _relative_error(output, expected) < _TOLERANCE, f"rank {rank}"


def test_an_attended_pad_would_be_caught(monkeypatch, reference):
    # 5 + 64 rows on 8 ranks: a pad of 3.
    inputs = _inputs(reference.zero_cond_t, text_len=5)
    (expected,) = _run(monkeypatch, reference, inputs, sp_size=1)
    (output, *_) = _run(monkeypatch, reference, inputs, sp_size=8, trim_pad=False)
    assert _relative_error(output, expected) > 10 * _TOLERANCE


def test_controlnet_residuals_are_refused_under_sp(monkeypatch, reference):
    monkeypatch.setattr(transformer_qwen, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(transformer_qwen, "get_sequence_parallel_world_size", lambda: 2)
    model = xFuserQwenImageTransformerWrapper(**_TINY_CONFIG, zero_cond_t=reference.zero_cond_t)
    inputs = _inputs(reference.zero_cond_t, text_len=4)
    residuals = [torch.zeros(_BATCH, inputs["hidden_states"].shape[1], 32)]
    with pytest.raises(NotImplementedError, match="ControlNet"):
        model(**inputs, controlnet_block_samples=residuals)
