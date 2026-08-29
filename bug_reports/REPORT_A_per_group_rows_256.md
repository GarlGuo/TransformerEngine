# Fused grouped MLP: per-group row counts that are not multiples of 256 silently corrupt everything

*Draft bug report for upstream (NVIDIA/TransformerEngine); not yet filed.*

## Title

`[PyTorch] GroupedMLP_CuTeGEMM: per-group row counts not divisible by 256 silently corrupt forward
and all gradients; only total rows are validated, and only against 128`

## Summary

The fused CuTeDSL grouped MLP requires **each group's row count** to be a multiple of 256. TE
validates only the **total** row count, and only against 128 (`grouped_mlp.py:1016`,
`if in_shape[0] % 128 != 0`). Between those two facts is a regime that passes every check and is
silently wrong:

> `sum(split_sizes) % 256 == 0` **and** some group's row count `≡ 128 (mod 256)`

In that regime nothing raises and nothing warns. The forward output, the input gradient, the
activation-scale gradient and both weight gradients are all corrupted — relative errors of order 1
against an fp32 reference, where the MXFP8 noise floor for these shapes is ~0.06.

This is **not** version-specific: TE 2.18.0 and 2.19.0.dev0 behave identically on 18 of 19 cells
tested, and it is still present at `origin/main` (`5ace9a31`). It needs no capacity tail — it
reproduces with `sum(split_sizes) == input.shape[0]`.

An MoE expert-parallel dispatch produces arbitrary per-expert token counts, so this is reachable in
ordinary training: a single expert receiving 258 rows into a 768-row capacity buffer silently
corrupts its weight gradients (measured below).

## Environment

| | |
|---|---|
| Affected | TransformerEngine `2.18.0` (`v2.18` = `27486e03`) **and** `2.19.0.dev0` @ `d8815c4c` |
| Also at | `origin/main` tip `5ace9a31` |
| GPU | NVIDIA GB200, SM 10.0 |
| torch | `2.12.0a0+0291f960b6.nv26.04` (NGC 26.04) |
| cuDNN FE | 1.26.0 |
| Recipe | `MXFP8BlockScaling` |
| Env | `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`, set before `import transformer_engine.pytorch` |

## Minimal reproducer

`num_groups=2`, 128 rows per group, buffer of 256 rows. **No tail** — `sum(split_sizes)` is the whole
buffer — and the total, 256, is a perfect multiple of 256. Both existing checks pass.

```python
import os
os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")  # read at TE import time

import torch
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import autocast, ops as te_ops

DEVICE, DTYPE = torch.device("cuda", 0), torch.bfloat16
HIDDEN, FFN = 1024, 2048
NUM_GROUPS, PER_GROUP, CAPACITY = 2, 128, 256
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

Relative Frobenius error vs the fp32 reference, printed by the script above. It exits 0 and raises
nothing on either version:

```
                                 forward       dX    dprob  FC1 wgrad  FC2 wgrad
TE 2.19.0.dev0  fused            0.53458  0.48984  0.74651    0.73533    0.88354
TE 2.19.0.dev0  unfused control  0.05616  0.06356  0.06096    0.06450    0.05613
TE 2.18.0       fused            0.52893  0.46075  0.91892    0.72333    0.94687
TE 2.18.0       unfused control  0.05619  0.06355  0.06140    0.06453    0.05619
```

Every quantity is corrupted, not just the weight gradients, and the two versions are equally
affected. Setting `NVTE_CUTEDSL_FUSED_GROUPED_MLP=0` returns every number to the MXFP8 floor, so the
fp32 reference is not in question.

Other shapes in the same regime (separate sweep, different seed, `tail=zero`, TE 2.19):

| cell | forward | dX | dprob | FC1 wgrad | FC2 wgrad |
|---|---|---|---|---|---|
| nG=4, 128/group, 512 rows | 0.942 | 1.104 | 1.200 | 0.974 | nan |
| nG=2, 384/group, 768 rows | 0.296 | 0.584 | 0.384 | 0.482 | 0.513 |

Every one of these has `sum(split_sizes) % 256 == 0` and `sum(split_sizes) == input.shape[0]`.

## Exact boundary

Sweep over group counts and per-group sizes, `hidden=1024 ffn=2048`, `used == capacity` (no tail),
both TE versions:

| condition | behaviour |
|---|---|
| every group's rows `% 256 == 0` | correct |
| `sum(split_sizes) % 256 != 0` | **raises** `ValueError: Invalid a.shape[0]` (CuTeDSL binding) |
| some group's rows `% 128 != 0` | **raises** device assert `grouped_layout.cuh:214`, then `CUDA_ERROR_ILLEGAL_INSTRUCTION` — also fires in the unfused path, so these sizes are outside the supported envelope generally |
| `sum % 256 == 0` **and** some group `≡ 128 (mod 256)` | **silently wrong** |

The dangerous regime is exactly the gap between the two loud checks.

At `num_groups == 1` without a tail, `total == per_group`, so every violation raises. The silent
regime is reachable at `num_groups == 1` only when `input.shape[0] > sum(split_sizes)` supplies a
legal `a.shape[0]` — which is what an MoE fixed-capacity receive buffer does:

| cell | TE 2.18 FC1 / FC2 wgrad | TE 2.19 FC1 / FC2 wgrad |
|---|---|---|
| nG=1, **258** rows used of 768 | 0.502 / 0.504 | 0.877 / 0.889 |
| nG=1, **384** rows used of 768 | 0.413 / 0.415 | 0.869 / 0.882 |

(The unfused control for `nG=1, 258 of 768` is 0.064 / 0.055.)

## Version delta

Identical on 18 of 19 cells — same correct set, same raising set, same silent set. The single
difference is 2.19 being worse: at `nG=2, 384/group` 2.19 also corrupts the **forward** (0.296)
where 2.18's forward stays at the floor (0.056) with only dX/dprob/wgrads broken.

Because the defect is version-independent, a 2.18-vs-2.19 A/B will not surface it.

## Why the test suite does not catch this

Every fused grouped-MLP test hardcodes `split_alignment: int = 256` and builds splits as `256*i` or
`256*(i+1)` (`tests/pytorch/test_grouped_mlp.py:946, 1346, 1436, 1698, 1901, 2003, 2150`), so a
violating size is never generated. `group_size` is never parametrized anywhere in
`test_grouped_mlp.py`, `test_grouped_linear.py` or `test_fusible_ops.py` — it is a keyword default.

The 256 requirement is documented nowhere a user would see it: `validate_grouped_mlp_dims`
(`grouped_mlp.py:756-788`) checks only `in_features`/`out_features % 64`, the glu interleave size and
FC1/FC2 agreement; the literal `256` does not appear in `grouped_mlp.py` at all. (The release-note
item "lowered weight dimension requirements to divisible by 64 (previously 256)" is about
`in_features`/`out_features`, not rows.)

## Suggested fix

1. Validate per-group row counts where `in_shape[0] % 128` is checked today (`grouped_mlp.py:1016`)
   or in `validate_grouped_mlp_dims`, and raise with the actual requirement. This closes the silent
   regime immediately.
2. Document the constraint on `GroupedLinear` / the fused op, since callers with MoE routing cannot
   generally satisfy it without padding each expert's rows to 256.
3. Longer term, support arbitrary per-group counts — MoE dispatch produces them by construction.

Add a regression test with `split_alignment=128` and a `group_size` parametrization; both are
currently unreachable configurations in the suite.
