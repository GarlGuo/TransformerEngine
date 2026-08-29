# Fused grouped MLP: at `num_groups == 1` the MXFP8 columnwise scale layout is misread, corrupting wgrad

*Draft bug report for upstream (NVIDIA/TransformerEngine); not yet filed.*

## Title

`[PyTorch] GroupedMLP_CuTeGEMM: at num_groups == 1 with input.shape[0] > sum(split_sizes), MXFP8
weight gradients are wrong — dense-quantized and kernel-packed columnwise scales share one
logical_shape`

## Summary

At `num_groups == 1`, TE builds the fused grouped MLP's operands two different ways and then
describes both with a single row extent:

- **Caller-supplied tensors** (the input, and `grad_output`) go through
  `_group_quantize_for_grouped_mlp:174`, which for `num_groups == 1` runs a **dense** `tex.quantize`
  over the *whole* buffer. Their columnwise scales are laid out for `input.shape[0]` rows.
- **Kernel-produced intermediates** (the FC1 activation output, the dgrads) come out of the fused
  CuTeDSL kernels packed for **`sum(m_splits)`** rows — stated explicitly in the layout comment at
  `grouped_mlp.py:1501-1506`.

Both are wrapped in a `GroupedTensor` whose `logical_shape` is the full buffer
(`_wrap_single_quantized_as_grouped:139`, `m_dim = tensor.shape[0]`; `:1590-1601`,
`shape=(in_shape[0], …)`), with the real extent carried only in `first_dims`. Every wgrad consumer
then reads *both* operands under one row extent, and misreads whichever one disagrees.

Because the columnwise swizzled scale layout is `[k/128][m/128][32][4][4]` — **m-tiles are an inner
stride** — reading 6 m-tiles where the producer wrote 2 puts every k-tile after the first at the
wrong offset: a reader fetching k-tile `j` lands on what the producer wrote for k-tile `3j`. Only
k-tile 0 aligns. The head rows' *data* is read correctly; only their scale factors come from the
wrong place, which is why the error is order 1 rather than subtly off.

The rowwise layout is `[m/128][k/128][32][4][4]` — m-tiles **outer** — so packed and dense coincide
there. The forward (`_single_group_fc2_gemm`, layout `TN`) and dgrad (`_single_group_dgrad_gemm`,
layout `NN`) consume rowwise scales and are unaffected. The wgrad is the only GEMM consuming
columnwise scales, and it is the only one corrupted. **A training run therefore looks healthy on
loss while its expert weight gradients are garbage.**

Introduced by `3e7ae6c` ("Single group mxfp8 grouped mlp", PR #3267), which extended the
`num_groups == 1` dense-quantize shortcut to MXFP8 — `v2.18` gated it
`if num_groups != 1 or not isinstance(quantizer, NVFP4Quantizer)`, so MXFP8 took `tex.group_quantize`
and every operand was packed consistently.

## Environment

| | |
|---|---|
| Bad | TransformerEngine `2.19.0.dev0` @ `d8815c4c`; unchanged at `origin/main` `5ace9a31` |
| Good | TransformerEngine `2.18.0`, tag `v2.18` = `27486e03` |
| GPU | NVIDIA GB200, SM 10.0 |
| torch | `2.12.0a0+0291f960b6.nv26.04` (NGC 26.04) |
| cuDNN FE | 1.26.0 |
| Recipe | `MXFP8BlockScaling` |
| Env | `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`, set before `import transformer_engine.pytorch` |

## Minimal reproducer

TransformerEngine only — no framework integration, no distributed, one GPU. A fixed-capacity buffer
whose trailing rows belong to no group, which is what an MoE expert-parallel dispatch produces.

```python
import os
os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")  # read at TE import time

import torch
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import autocast, ops as te_ops

DEVICE, DTYPE = torch.device("cuda", 0), torch.bfloat16
HIDDEN, FFN = 1024, 2048
NUM_GROUPS, PER_GROUP, CAPACITY = 1, 256, 768
USED = NUM_GROUPS * PER_GROUP

torch.manual_seed(0)
fc1 = te_ops.GroupedLinear(NUM_GROUPS, HIDDEN, FFN, bias=False, device=DEVICE, dtype=DTYPE)
fc2 = te_ops.GroupedLinear(NUM_GROUPS, FFN, HIDDEN, bias=False, device=DEVICE, dtype=DTYPE)
ops = te_ops.Sequential(fc1, te_ops.ScaledSReLU(), fc2)
w1 = [getattr(fc1, f"weight{i}") for i in range(NUM_GROUPS)]
w2 = [getattr(fc2, f"weight{i}") for i in range(NUM_GROUPS)]

x = torch.randn(CAPACITY, HIDDEN, dtype=DTYPE, device=DEVICE).requires_grad_()
probs = (torch.rand(CAPACITY, dtype=torch.float32, device=DEVICE) + 0.5).requires_grad_()
grad_out = torch.randn(CAPACITY, HIDDEN, dtype=DTYPE, device=DEVICE)
splits = torch.full((NUM_GROUPS,), PER_GROUP, dtype=torch.int64, device=DEVICE)

with autocast(enabled=True, recipe=MXFP8BlockScaling()):
    out = ops(x, splits, probs, splits)
out.backward(grad_out)

# fp32 reference over the rows that belong to a group. Built AFTER the TE call on purpose:
# building it first perturbs the caching allocator and turns the corruption into NaN.
x32 = x.detach()[:USED].float().requires_grad_()
p32 = probs.detach()[:USED].float().requires_grad_()
r1 = [w.detach().float().requires_grad_() for w in w1]
r2 = [w.detach().float().requires_grad_() for w in w2]
chunks = []
for g in range(NUM_GROUPS):
    rows = slice(g * PER_GROUP, (g + 1) * PER_GROUP)
    h = torch.nn.functional.relu(x32[rows] @ r1[g].t())
    chunks.append((h * h * p32[rows].unsqueeze(-1)) @ r2[g].t())
torch.cat(chunks).backward(grad_out[:USED].float())

rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())
print(f"forward    {rel(out[:USED], torch.cat(chunks)):.5f}")
print(f"dX         {rel(x.grad[:USED], x32.grad):.5f}")
print(f"dprob      {rel(probs.grad[:USED], p32.grad):.5f}")
print(f"FC1 wgrad  {rel(torch.stack([w.grad for w in w1]), torch.stack([w.grad for w in r1])):.5f}")
print(f"FC2 wgrad  {rel(torch.stack([w.grad for w in w2]), torch.stack([w.grad for w in r2])):.5f}")
print("MXFP8 noise floor for this shape is ~0.06")
```

### Expected vs actual

Relative Frobenius error over the first `sum(split_sizes)` rows — the only rows that belong to a
group — against the fp32 reference, printed by the script above. The MXFP8 floor for this shape is
~0.06.

```
                                 forward    dgrad    dprob  FC1 wgrad  FC2 wgrad
TE 2.18.0       fused            0.05607  0.06305  0.06056    0.06416    0.05583
TE 2.19.0.dev0  fused            0.05607  0.06305  0.06056    0.89580    0.90810
TE 2.19.0.dev0  unfused control  0.05623  0.06310  0.06046    0.06426    0.05600
```

Forward, dgrad and dprob are **bit-for-bit identical** on the two versions. Only the weight gradients
differ. `NVTE_CUTEDSL_FUSED_GROUPED_MLP=0`, `num_groups >= 2`, and
`input.shape[0] == sum(split_sizes)` are all clean on both versions.

### A note on NaN

The corrupted weight gradient is often, but not reliably, NaN. Whether it is depends on the state of
the PyTorch caching allocator: the rows of TE's own intermediate buffers past `sum(split_sizes)` are
never written by the fused kernels, so they hold whatever was last in that memory. A script that
allocates and frees float32 tensors before the TE call (for example, one that builds its reference
first) tends to produce NaN; the script above, run on fresh pages, deterministically produces the
finite-but-wrong values shown. **Assert on the error magnitude, not on `isnan`** — an `isnan` check
prints `False` on an affected build about as often as `True`.

## Evidence that the tail's *contents* are not the mechanism

This matters for the fix, because the obvious reading — "the GEMM contracts over the unused tail, so
the tail's values land in the weight gradient" — is wrong.

1. **The tail cannot contribute.** Instrumenting `_single_group_wgrad_gemm` and dumping the raw
   MXFP8 bytes of its operands: in the FC2 wgrad the x operand's tail is 1,048,576 bytes of `0x00`;
   in the FC1 wgrad the dy operand's tail is 1,048,576 bytes of `0x00`. No `0xFF` E8M0 scale bytes
   anywhere. One side of each GEMM is identically zero over the tail, so the tail rows contribute
   exactly zero — yet FC1 is 0.896 and FC2 0.908 wrong.
2. **The produced buffer is packed, not sparse.** An occupancy map of the kernel-produced columnwise
   scale buffer (16 k-tiles × 6 m-tiles × 512 B) shows a **contiguous flat prefix**: k00–k04 all six
   m-slots, k05 the first two, k06+ empty. A genuine 768-row layout with only 256 rows written would
   show `X X . . . .` on *every* k-tile. It shows that on none.
3. **The prefix is the 256-row tensor.** It is bit-for-bit equal to the entire columnwise scale
   buffer from the `capacity=256` cell, and *not* equal to the dense-768 head slice.
4. **The data is fine.** Columnwise *data* head rows are bit-identical between the broken
   (`capacity=768`) and clean (`capacity=256`) cells for all four operands. Only the scales move.

## Where

`transformer_engine/pytorch/ops/fused/grouped_mlp.py` @ `2.19.0.dev0`:

- **`:174`** — the producer. `_group_quantize_for_grouped_mlp` takes `tex.quantize` (dense, whole
  buffer) for `num_groups == 1` + MXFP8/NVFP4. In `v2.18` this was gated to NVFP4 only.
- **`:139`** — `_wrap_single_quantized_as_grouped` sets `m_dim = tensor.shape[0]` while
  `first_dims=split_sizes` (`:158`), creating the mismatch.
- **`:1501-1506`** — the kernel's layout comment, which states the produced extent is
  `sum(m_splits)`, next to `:1590-1601` which declares `shape=(in_shape[0], …)`. Same pattern for the
  backward's dgrad tensors at `:2212`, `:2280`, `:2348`.
- **`:269`** — the consumer. `_single_quantized_tensor_from_grouped` does
  `shape = tuple(grouped.logical_shape)`, entering the columnwise view at `:292`
  (`quantizer.get_scale_shape(shape, True)`). This line predates `3e7ae6c`.
- **`:688-699`** — the `num_groups == 1` wgrad dispatch that reaches it.
- **`:480`, `:533`, `:544`** — `_cudnn_compute_wgrad` takes `total_tokens =
  grouped_dy.logical_shape[0]` and views the columnwise scales at `round_up(in_features, 128)`, i.e.
  it makes the *same* assumption. It is not a safe fallback (see below).

## What a fix has to do

Measured on the failing cell, TE 2.19, patching each `num_groups == 1` fast path individually
(`q` = force `tex.group_quantize`; `w` = route the wgrad away from `_single_group_wgrad_gemm`;
`f2`/`dg` = route the FC2 forward / dgrad to the grouped kernels):

| patch | FC1 wgrad | FC2 wgrad | |
|---|---|---|---|
| none | 0.88640 | nan | baseline |
| `w` | 0.27179 | 0.28838 | partial |
| `q` | **nan** | nan | **worse than baseline** |
| `f2` | 0.88640 | nan | no effect |
| `dg` | 0.88640 | nan | no effect |
| `w,f2,dg` | 0.27179 | 0.28838 | = `w` |
| **`q,w`** | **0.06416** | **0.05583** | **matches TE 2.18 exactly** |

Three consequences:

- **Gating the `:688` dispatch and falling through to cuDNN is not sufficient** — that is `w`,
  0.272/0.288, still ~4.5× the floor and 4× worse than 2.18. `_cudnn_compute_wgrad` carries the same
  `logical_shape`-not-`first_dims` assumption.
- **Fixing only the producer is worse than doing nothing** (`q` → NaN): it leaves a dense-view
  consumer reading a packed producer.
- **Slicing the views to `first_dims[0]` is wrong as a blanket fix.** The caller-quantized operands
  genuinely *are* in dense layout for `logical_shape[0]` rows; slicing them would just move the
  corruption to the other operand. (It is also not a contiguous view for the columnwise swizzled
  scales.)

The durable fix is to make the two extents distinguishable rather than to pick one: have the
producer declare `sum(split_sizes)` when that is what it wrote, or give `GroupedTensor` an explicit
scale-layout extent so consumers cannot silently assume `logical_shape`.

## Why the test suite does not catch this

The two ingredients are each covered, never together:

- `test_grouped_mlp_single_group_mxfp8` (`tests/pytorch/test_grouped_mlp.py:1570-1599`) delegates to
  `test_grouped_mlp(group_size=1, …)`, which builds `in_shape = (split_sizes.sum().item(),
  hidden_size)` (`:961`) — **exactly `sum(split_sizes)`, never a tail**.
- A tail *is* covered, at `group_size=4`, by `test_grouped_mlp_cuda_graph_safe_mxfp8` (`:2140-2404`,
  `token_padding=2048`), and it does compare weight grads unsliced (`:2402+`). But its reference is
  **another TE module running the same fused path on the same padded input** (`:2381`), so both sides
  make the identical error and the test passes. The comments at `:907` / `:2394` note the padded tail
  "remains uninitialized" and slice it out of the output/dx checks; the wgrad implication was not
  considered.

A `group_size=1` case with `token_padding > 0`, checked against a non-TE reference, would fail today.
