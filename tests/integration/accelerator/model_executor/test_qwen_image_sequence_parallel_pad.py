"""Qwen-Image under Ulysses and ring attention matches one GPU, also for a prompt shorter than the SP degree.

Each rank attends over its text rows followed by its image rows. The streams are split unevenly
(TwoStreamSPSplit), so that every rank holds as many rows and the only SP pad trails the gathered keys, where USP
drops it given the real length (`valid_kv_len`). This compares the outputs with a one-GPU run, which has no pad.
Tiny random weights; nothing is downloaded.

Random weights draw no attention sink, so an attended pad errs by only 1e-4 to 8e-4 here, which bf16 GEMMs and
norms would hide; the model therefore runs in fp32. Attention runs through PyTorch's SDPA entry point, except under
ring attention, whose merge needs the log-sum-exp that only SDPA_EFFICIENT returns. FP8 comms needs a kernel that
takes pre-quantized FP8 inputs, AITER_FP8, and runs only under Ulysses, so its one-GPU reference has the test wrap
the kernel in the steps FP8 comms takes around it, as test_wan_sequence_parallel_pad.py does. A bf16 model, the only
one to run the fused QK-norm/RoPE kernel, checks the ranks that hold no text rows, also with FP8 GEMMs, which raise on
the empty text stream unless it skips them. Edit-2511's zero_cond_t splits its
per-token modulation index along with the image. Every run also checks that the last rank keeps its pad rows at
zero in every block's image output.
"""

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from _spawn import spawn_accelerator_ranks
from test_wan_sequence_parallel_pad import _attend_as_fp8_comms

pytestmark = [pytest.mark.multi_gpu, pytest.mark.slow]

_TINY_CONFIG = dict(
    patch_size=2,
    in_channels=16,
    out_channels=4,
    num_layers=2,
    attention_head_dim=128,
    # Divisible by the Ulysses degree, up to 8.
    num_attention_heads=8,
    joint_attention_dim=64,
    axes_dims_rope=(16, 56, 56),
)
# 64 image tokens, so the pad (fewer rows than the SP degree) fits on the last of 8 ranks: one image, or for
# zero_cond_t the edited image followed by a condition image.
_IMG_SHAPES = {"qwen-image": [(1, 8, 8)], "zero-cond-t": [(1, 6, 8), (1, 4, 4)]}
# 3 + 64 rows: no text rows on some ranks of 4 or 8, a pad of 1 on 2 and 4 ranks, 5 on 8.
SHORT_PROMPT = 3
# 4 + 64 rows split evenly on 2 and 4 ranks.
UNPADDED_PROMPT = 4
MODES = {
    "ulysses": (2, 1),
    "ring": (1, 2),
    "ulysses-ring": (2, 2),
    "ulysses8": (8, 1),
    "ulysses4-ring2": (4, 2),
}


class Numerics(NamedTuple):
    backend: str
    dtype: torch.dtype
    fp8_comms: bool = False
    fp8_gemms: bool = False


NUMERICS = {
    "fp32-sdpa": Numerics("SDPA", torch.float32),
    "fp32-sdpa-efficient": Numerics("SDPA_EFFICIENT", torch.float32),
    "fp8-comms": Numerics("AITER_FP8", torch.float32, fp8_comms=True),
    "bf16": Numerics("SDPA", torch.bfloat16),
    "fp8-gemms": Numerics("SDPA", torch.bfloat16, fp8_gemms=True),
}
# Largest relative L2 error against one GPU. Measured with the pad dropped / attended: in fp32 <= 3.2e-7 / >= 1.2e-4,
# with FP8 comms <= 3.3e-5 / >= 3.7e-4. The bf16 model rounds by ~9e-4 with the pad dropped and ~2e-3 with it
# attended, so it only catches gross errors, such as rows lost on the ranks without text. FP8 GEMMs round by ~3.7e-3
# either way.
TOLERANCE = {"fp32-sdpa": 1e-5, "fp32-sdpa-efficient": 1e-5, "fp8-comms": 1e-4, "bf16": 5e-3, "fp8-gemms": 2e-2}
# A fixed scale, so the quantization does not depend on the data, whose pad rows differ between the runs.
FP8_COMMS_SCALE = 0.02


def _inputs(variant, text_len, batch):
    generator = torch.Generator().manual_seed(1)
    shapes = _IMG_SHAPES[variant]
    image_len = sum(f * h * w for f, h, w in shapes)
    return dict(
        hidden_states=torch.randn(batch, image_len, _TINY_CONFIG["in_channels"], generator=generator),
        encoder_hidden_states=torch.randn(batch, text_len, _TINY_CONFIG["joint_attention_dim"], generator=generator),
        timestep=torch.full((batch,), 0.5),
    ), [shapes] * batch


def _watch_pad_rows(model, rows):
    """Record, for every block, whether the last `rows` rows of its image output (this rank's SP pad) are zero."""
    zero = []

    def record(module, args, output):
        zero.append(bool(output[1][:, -rows:].eq(0).all()))

    for block in model.transformer_blocks:
        block.register_forward_hook(record)
    return zero


def _worker(rank, world_size, init_method, variant, ulysses, ring, text_len, numerics, batch, output_path):
    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        get_sequence_parallel_rank,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.models.transformers.transformer_qwen import xFuserQwenImageTransformerWrapper

    numerics = NUMERICS[numerics]
    fp8_comms = numerics.fp8_comms and ulysses > 1
    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=ulysses, ring_degree=ring)
    try:
        args = xFuserArgs(
            attention_backend=numerics.backend,
            ulysses_degree=ulysses,
            ring_degree=ring,
            use_fp8_comms=fp8_comms,
            fp8_comms_scale=FP8_COMMS_SCALE if fp8_comms else None,
        )
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        runtime_state = get_runtime_state()
        runtime_state.set_attention_backend(numerics.backend)
        if numerics.fp8_comms and not fp8_comms:
            _attend_as_fp8_comms(numerics.backend)

        device = torch.device("cuda", rank)
        # Seeded on the CPU, so every process builds the same weights.
        torch.manual_seed(0)
        model = xFuserQwenImageTransformerWrapper(**_TINY_CONFIG, zero_cond_t=variant == "zero-cond-t")
        model = model.eval().to(device=device, dtype=numerics.dtype)
        if numerics.fp8_gemms:
            from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp8

            quantize_linear_layers_to_fp8(model.transformer_blocks)
        if fp8_comms:
            runtime_state.fp8_comms.register_models(SimpleNamespace(transformer=model))

        tensors, img_shapes = _inputs(variant, text_len, batch)
        sp_size = ulysses * ring
        pad_rows = -(text_len + tensors["hidden_states"].shape[1]) % sp_size
        sp_rank = get_sequence_parallel_rank()
        pad_rows_zero = _watch_pad_rows(model, pad_rows) if pad_rows and sp_rank == sp_size - 1 else None

        inputs = {
            key: tensor.to(device=device, dtype=tensor.dtype if key == "timestep" else numerics.dtype)
            for key, tensor in tensors.items()
        }
        with torch.no_grad():
            output = model(**inputs, img_shapes=img_shapes, return_dict=False)[0]

        if pad_rows_zero is not None:
            assert pad_rows_zero and all(pad_rows_zero), (
                f"SP rank {sp_rank}: its {pad_rows} pad rows are non-zero in "
                f"{pad_rows_zero.count(False)} of {len(pad_rows_zero)} block outputs"
            )
        if rank == 0:
            torch.save(output.cpu(), output_path)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _run(directory, variant, ulysses, ring, text_len, numerics, batch=1):
    output_path = directory / "output.pt"
    spawn_accelerator_ranks(
        _worker,
        directory,
        world_size=ulysses * ring,
        timeout=600,
        args=(variant, ulysses, ring, text_len, numerics, batch, str(output_path)),
    )
    return torch.load(output_path)


@pytest.fixture(scope="module")
def one_gpu_output(tmp_path_factory):
    """The one-GPU output for a variant, prompt length, numerics and batch size, run once per module."""
    outputs = {}

    def get(variant, text_len, numerics, batch=1):
        key = (variant, text_len, numerics, batch)
        if key not in outputs:
            outputs[key] = _run(tmp_path_factory.mktemp("one-gpu"), variant, 1, 1, text_len, numerics, batch)
        return outputs[key]

    return get


def _skip_unless_available(numerics, ring=1):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType

    spec = registry.get(AttentionBackendType[NUMERICS[numerics].backend])
    reason = spec.unavailable() or (spec.ring.unmet() if ring > 1 else None)
    if reason is not None:
        pytest.skip(reason)


def _relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _fp32(mode):
    """The fp32 numerics for a mode: PyTorch's SDPA, unless a ring merge needs the log-sum-exp it does not return."""
    return "fp32-sdpa-efficient" if MODES[mode][1] > 1 else "fp32-sdpa"


def _assert_matches_one_gpu(tmp_path, one_gpu_output, variant, mode, text_len, numerics, batch=1):
    ulysses, ring = MODES[mode]
    _skip_unless_available(numerics, ring)
    expected = one_gpu_output(variant, text_len, numerics, batch)

    actual = _run(tmp_path, variant, ulysses, ring, text_len, numerics, batch)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE[numerics], f"relative error {error:.3e}"


@pytest.mark.parametrize("mode", ["ulysses", "ulysses-ring"])
def test_unpadded_sequence_parallel_matches_one_gpu(tmp_path, one_gpu_output, mode):
    _assert_matches_one_gpu(tmp_path, one_gpu_output, "qwen-image", mode, UNPADDED_PROMPT, _fp32(mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["qwen-image", "zero-cond-t"])
def test_short_prompt_matches_one_gpu(tmp_path, one_gpu_output, variant, mode):
    _assert_matches_one_gpu(tmp_path, one_gpu_output, variant, mode, SHORT_PROMPT, _fp32(mode))


def test_batch_of_two_matches_one_gpu(tmp_path, one_gpu_output):
    """Different samples per batch row, under both exchanges and the ring merge, and through the uneven gather."""
    mode = "ulysses-ring"
    _assert_matches_one_gpu(tmp_path, one_gpu_output, "zero-cond-t", mode, SHORT_PROMPT, _fp32(mode), batch=2)


@pytest.mark.parametrize("mode", ["ulysses", "ulysses8"])
def test_short_prompt_with_fp8_comms_matches_one_gpu(tmp_path, one_gpu_output, mode):
    """AITER_FP8 joins no ring, so Ulysses only."""
    _assert_matches_one_gpu(tmp_path, one_gpu_output, "qwen-image", mode, SHORT_PROMPT, "fp8-comms")


@pytest.mark.parametrize("numerics", ["bf16", "fp8-gemms"])
@pytest.mark.parametrize("mode", ["ulysses", "ulysses8"])
def test_bf16_model_runs_ranks_without_text_rows(tmp_path, one_gpu_output, mode, numerics):
    """A one-row prompt leaves the second of two ranks without text rows, a three-row one five of eight."""
    if NUMERICS[numerics].fp8_gemms:
        pytest.importorskip("torchao")
    text_len = 1 if mode == "ulysses" else SHORT_PROMPT
    _assert_matches_one_gpu(tmp_path, one_gpu_output, "qwen-image", mode, text_len, numerics)
