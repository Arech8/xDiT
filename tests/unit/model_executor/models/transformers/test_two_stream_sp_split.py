"""TwoStreamSPSplit gives every rank a joint `[first, last]` chunk of one length, whose only SP pad trails the keys.

USP gathers the ranks' chunks in rank order and drops a trailing pad given `valid_kv_len`; a pad anywhere else would
draw attention from every real token. Padding each stream separately would put pad after every rank's `first` rows.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.models.transformers import transformers_utils
from xfuser.model_executor.models.transformers.transformers_utils import (
    TwoStreamSPSplit,
    skip_empty_sequence,
    zero_sp_pad_rows_hook,
)

# Long enough that the pad fits on the last rank: more rows than sp_size * (sp_size - 1).
CASES = [
    (first_len, last_len, sp_size)
    for sp_size in (1, 2, 3, 4, 8)
    for first_len in (0, 1, 2, 5, 7, 8, 9, 13, 64, 300)
    for last_len in (57, 100, 4096)
]


def _ids(start, length):
    # One-based, so that the zero pad stands out.
    return torch.arange(start + 1, start + length + 1)


@pytest.mark.parametrize("first_len, last_len, sp_size", CASES)
def test_the_gathered_chunks_hold_both_streams_in_order_and_the_pad_last(first_len, last_len, sp_size):
    split = TwoStreamSPSplit.plan(first_len, last_len, sp_size)
    first, last = _ids(0, first_len), _ids(first_len, last_len)
    chunks = [torch.cat([split.shard_first(first, r, 0), split.shard_last(last, r, 0)]) for r in range(sp_size)]

    assert {len(chunk) for chunk in chunks} == {split.local_len}
    assert 0 <= split.pad < sp_size
    gathered = torch.cat(chunks)
    real_len = first_len + last_len
    assert split.valid_kv_len == (real_len if split.pad else None)
    assert gathered[real_len:].eq(0).all() and gathered[:real_len].ne(0).all()
    real = gathered[:real_len]
    assert torch.equal(real[real <= first_len], first)
    assert torch.equal(real[real > first_len], last)
    # Neither stream's rows pile up on a few ranks.
    assert max(split.first_rows) - min(split.first_rows) <= 1
    assert [split.local_pad_rows(r) for r in range(sp_size)] == [0] * (sp_size - 1) + [split.pad]


def test_a_pad_longer_than_the_last_ranks_share_of_last_takes_first_rows_elsewhere():
    # An even split would leave the last rank 2 rows of `first` and room for 1 of `last`, short of the 3 pad rows.
    split = TwoStreamSPSplit.plan(8, 1, 4)
    assert split == TwoStreamSPSplit(first_rows=(3, 3, 2, 0), last_rows=(0, 0, 1, 0), pad=3)


def test_a_pad_that_does_not_fit_on_the_last_rank_is_refused():
    with pytest.raises(ValueError, match="too short"):
        TwoStreamSPSplit.plan(7, 2, 8)


def _gather_on_every_rank(monkeypatch, gather, chunks):
    """`gather(chunk, rank)` on every rank, against a stand-in SP group.

    A first pass records what each rank sends, a second hands every rank all of it.
    """
    sent = []

    def record(x, dim):
        sent.append(x)
        return torch.cat([x] * len(chunks), dim)

    monkeypatch.setattr(transformers_utils, "get_sp_group", lambda: SimpleNamespace(all_gather=record))
    for rank, chunk in enumerate(chunks):
        gather(chunk, rank)
    assert len({x.shape for x in sent}) <= 1, "the collective needs equal shapes"
    monkeypatch.setattr(
        transformers_utils, "get_sp_group", lambda: SimpleNamespace(all_gather=lambda x, dim: torch.cat(sent, dim))
    )
    return [gather(chunk, rank) for rank, chunk in enumerate(chunks)]


@pytest.mark.parametrize("first_len, last_len, sp_size", [(7, 57, 8), (8, 1, 4), (300, 100, 3), (5, 100, 1)])
def test_either_stream_gathers_back_whole_from_its_uneven_shards(monkeypatch, first_len, last_len, sp_size):
    split = TwoStreamSPSplit.plan(first_len, last_len, sp_size)
    # A batch of two, so the gather runs along a dim other than the leading one.
    first, last = torch.randn(2, first_len, 3), torch.randn(2, last_len, 3)

    first_chunks = [split.shard_first(first, r, dim=1) for r in range(sp_size)]
    for gathered in _gather_on_every_rank(monkeypatch, lambda x, r: split.gather_first(x, r, dim=1), first_chunks):
        assert torch.equal(gathered, first)

    last_chunks = [split.shard_last(last, r, dim=1) for r in range(sp_size)]
    assert not last_chunks[-1][:, last_chunks[-1].shape[1] - split.pad :].any()
    for gathered in _gather_on_every_rank(monkeypatch, lambda x, r: split.gather_last(x, r, dim=1), last_chunks):
        assert torch.equal(gathered, last)


def test_the_hook_zeroes_the_pad_rows_of_the_indexed_stream_only():
    text, image = torch.ones(1, 4, 2), torch.ones(1, 6, 2)
    zero_sp_pad_rows_hook({"local_rows": 2}, index=1)(None, (), (text, image))
    assert text.eq(1).all()
    assert not image[:, -2:].any()
    assert image[:, :-2].eq(1).all()


def test_a_module_skipping_empty_sequences_still_runs_on_rows_and_survives_a_deepcopy():
    module = torch.nn.Linear(3, 3)
    skip_empty_sequence(module)
    empty, rows = torch.ones(2, 0, 3), torch.ones(2, 4, 3)
    assert module(empty) is empty
    copy = deepcopy(module)
    with torch.no_grad():
        copy.weight.zero_()
        copy.bias.zero_()
    # A forward bound to the original would still compute with its weights.
    assert not copy(rows).any()
    assert module(rows).any()
