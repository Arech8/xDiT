import dataclasses
import types

import torch
from xfuser.core.distributed import (
    get_sp_group,
)

def chunk_and_pad_sequence(x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int) -> torch.Tensor:
    if pad_amount > 0:
        if dim < 0:
            dim = x.ndim + dim
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_amount
        x = torch.cat([x,
                    torch.zeros(
                        pad_shape,
                        dtype=x.dtype,
                        device=x.device,
                    )], dim=dim)
    x = torch.chunk(x,
                    sp_world_size,
                    dim=dim)[sp_world_rank]
    return x

def gather_and_unpad(x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
    x = get_sp_group().all_gather(x, dim=dim)
    size = x.size(dim)
    return x.narrow(dim=dim, start=0, length=size - pad_amount)


def local_sp_pad_rows(seq_len: int, local_len: int, sp_rank: int) -> int:
    """How many of this rank's `local_len` rows are SP pad, for a real length `seq_len`.

    The pad trails the global sequence, so only the last ranks' chunks hold any.
    """
    return min(local_len, max(0, (sp_rank + 1) * local_len - seq_len))


def zero_sp_pad_rows_hook(sp_pad: dict, index: int | None = None):
    """Forward hook keeping the SP pad rows of a block's output at zero.

    `sp_pad["local_rows"]` counts the pad rows ending this rank's chunk along
    dim 1; the model sets it per forward.

    The pad rows never reach real tokens' attention once USP drops their keys,
    but what a block writes there still enters statistics taken across the
    rows of whole local tensors, which then apply to the real rows:
    - FP8-comms calibration: pad rows set the per-tensor V scale in some layers
      (measured on Wan up to 1.5x coarser for every real token);
    - the step cache's residual difference;
    - attention kernels that share a quantization scale across query rows:
      Sage (256-row Q blocks) and AITER FP8 (one Q scale per tensor).
    The list spans features and backends, so the zeroing is not gated on them;
    missing one would fail silently. It only runs on a rank holding pad rows,
    and costs nothing measurable on Wan: under torch.compile it fuses into the
    block's last pointwise op, eagerly it is ~5 us per block.

    A hook rather than a line in forward's loop, so it also runs when a step
    cache iterates the blocks itself. Of a block returning a tuple, every
    tensor is zeroed unless `index` picks the one holding the pad: a VACE
    block's conditioning states are added to the main stream as hints, so they
    are zeroed with its control states, while a dual-stream block's other
    stream holds no pad.
    """

    def hook(module, args, output):
        rows = sp_pad["local_rows"]
        if rows:
            tensors = output if isinstance(output, tuple) else (output,)
            if index is not None:
                tensors = (tensors[index],)
            for tensor in tensors:
                if tensor is not None:
                    tensor[:, -rows:] = 0

    return hook


def _forward_or_empty(self, hidden_states: torch.Tensor, *args, **kwargs):
    if hidden_states.shape[1] == 0:
        return hidden_states
    return type(self).forward(self, hidden_states, *args, **kwargs)


def skip_empty_sequence(module: torch.nn.Module) -> None:
    """Make `module` return its input as is when that holds no rows along dim 1.

    Under TwoStreamSPSplit a rank may hold none of a short stream, and quantized
    GEMMs raise on an empty input (torchao's dynamic FP8 takes an amax over the
    rows). Only for a module whose output has its input's shape, e.g. a
    FeedForward. Set on the module rather than on its Linear layers, which the
    GEMM quantizers may swap after construction; its parameters and state_dict
    keys stay as they are. Bound with MethodType rather than a closure, so that
    a deepcopy of the model calls the copy's own forward.
    """
    module.forward = types.MethodType(_forward_or_empty, module)


def _append_zero_rows(x: torch.Tensor, rows: int, dim: int) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = rows
    return torch.cat([x, x.new_zeros(shape)], dim=dim)


def _all_gather_uneven(x: torch.Tensor, rows: tuple[int, ...], rank: int, dim: int) -> torch.Tensor:
    """Concatenate, in SP rank order, the leading `rows[r]` rows along `dim` of every rank's `x`.

    The collective needs equal shapes, so each chunk travels padded to the longest.
    """
    x = x.narrow(dim, 0, rows[rank])
    if len(rows) == 1:
        return x
    width = max(rows)
    if rows[rank] < width:
        x = _append_zero_rows(x, width - rows[rank], dim)
    chunks = get_sp_group().all_gather(x.contiguous(), dim=dim).split(width, dim=dim)
    return torch.cat([chunk.narrow(dim, 0, n) for chunk, n in zip(chunks, rows)], dim=dim)


@dataclasses.dataclass(frozen=True)
class TwoStreamSPSplit:
    """Sequence-parallel split of a joint attention sequence made of two streams.

    Each rank concatenates its rows of the two streams as `[first, last]`
    before attention. Padding each stream to a multiple of the SP degree, as
    chunk_and_pad_sequence does for one, would leave pad rows inside the
    gathered keys, after every rank's share of `first`. USP's `valid_kv_len`
    only drops a trailing pad, so real tokens would attend to those rows.

    Instead every rank holds the same number of joint rows, `local_len`, and
    the streams are split unevenly: `first` as evenly as the pad allows, `last`
    filling the rest. The only pad, fewer than `sp_size` rows, ends the last
    rank's `last` rows. Every USP layout gathers the chunks in rank order
    (ring shards are contiguous in it too), so the pad trails the keys and
    `valid_kv_len` drops it, and no stream has to be replicated, which keeps
    FP8 comms (joint-tensor-free USP calls only) usable.

    Ranks still do equal work when both streams cost the same per row, as
    Qwen-Image's equally sized text and image layers do. A stream shorter than
    `sp_size` leaves some ranks without any of its rows; they run that
    stream's layers on empty tensors.
    """

    # Rows of the stream concatenated first, per rank.
    first_rows: tuple[int, ...]
    # Real rows of the stream concatenated last, per rank. The last rank's are followed by `pad` zero rows.
    last_rows: tuple[int, ...]
    pad: int

    @classmethod
    def plan(cls, first_len: int, last_len: int, sp_size: int) -> "TwoStreamSPSplit":
        if sp_size == 1:
            return cls((first_len,), (last_len,), 0)
        local_len = -(-(first_len + last_len) // sp_size)
        pad = sp_size * local_len - first_len - last_len
        if pad > local_len:
            raise ValueError(
                f"A joint sequence of {first_len + last_len} rows is too short for SP degree {sp_size}: "
                f"its pad of {pad} rows would not fit on the last rank, which holds {local_len}."
            )
        # The pad ends the last rank's chunk, which leaves room there for at most `local_len - pad` rows of `first`.
        last_rank_first = min(first_len // sp_size, local_len - pad)
        base, extra = divmod(first_len - last_rank_first, sp_size - 1)
        first_rows = tuple(base + (rank < extra) for rank in range(sp_size - 1)) + (last_rank_first,)
        last_rows = tuple(local_len - rows for rows in first_rows[:-1]) + (local_len - last_rank_first - pad,)
        return cls(first_rows, last_rows, pad)

    @property
    def local_len(self) -> int:
        return self.first_rows[-1] + self.last_rows[-1] + self.pad

    @property
    def valid_kv_len(self) -> int | None:
        """USP's `valid_kv_len` for the gathered keys; None without a pad, which skips USP's trim."""
        return sum(self.first_rows) + sum(self.last_rows) if self.pad else None

    def local_pad_rows(self, rank: int) -> int:
        """Pad rows ending this rank's `last` rows, as zero_sp_pad_rows_hook counts them."""
        return self.pad if rank == len(self.first_rows) - 1 else 0

    def shard_first(self, x: torch.Tensor, rank: int, dim: int) -> torch.Tensor:
        return x.narrow(dim, sum(self.first_rows[:rank]), self.first_rows[rank])

    def shard_last(self, x: torch.Tensor, rank: int, dim: int) -> torch.Tensor:
        x = x.narrow(dim, sum(self.last_rows[:rank]), self.last_rows[rank])
        pad = self.local_pad_rows(rank)
        return _append_zero_rows(x, pad, dim) if pad else x

    def gather_first(self, x: torch.Tensor, rank: int, dim: int) -> torch.Tensor:
        return _all_gather_uneven(x, self.first_rows, rank, dim)

    def gather_last(self, x: torch.Tensor, rank: int, dim: int) -> torch.Tensor:
        """The full `last` stream; the pad ending the last rank's rows is dropped."""
        return _all_gather_uneven(x, self.last_rows, rank, dim)
