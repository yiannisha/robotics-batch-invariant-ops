# LingBot-VLA 2.0 PyTorch investigation

## Scope

This experiment uses the authors' official PyTorch repository at commit
`be969b8`, official 6B RoboTwin checkpoint at revision `04518557`, and pinned
Qwen3-VL tokenizer/config assets. The checkpoint has 6,375,907,511 elements in
1,708 tensors and loads strictly without conversion. Inference runs the
official three-camera feature transform, state normalization, Qwen3-VL vision
and language prefix, 36-layer MoE action expert, and all ten flow-matching
steps. The robot output is the complete normalized 50x55 action chunk.

The input fixture is a real 396-frame, three-camera ALOHA-style episode with a
14-D state and task text. Its keys map directly through the official RoboTwin
robot configuration. This gives reproducible real sensor content but is not
used to claim RoboTwin task performance.

## Nondeterminism and upstream state controls

Every frame receives explicit fixed 50x55 flow noise. The adapter clones it
because the released `sample_actions` mutates its `noise` argument in place.
The official vision integration caches grid metadata from the first request;
the adapter clears those five shape-derived fields before each inference so a
B=1 request cannot poison a following B=2 request.

The released inference-only `robby_moe_forward` is not repeatable at B=1. Its
Triton route pack uses relaxed `atomic_add` to assign expert rows, and its down
projection uses relaxed `atomic_add` to accumulate top-k routes. Two identical
official-kernel B=1 calls differ in all 2,750 action values, with maximum
absolute difference `0.004252731800079346` in the recorded run.

The reusable `deterministic_token_choice_moe` fallback preserves the same
top-k SwiGLU semantics while avoiding atomic routing and accumulation. Three
fixed-schedule invariant BMMs evaluate all 32 experts, selected routes are
gathered, and the fixed top-k dimension is summed in a fixed order. Its unit
test matches an explicit route-by-route reference. At the complete-model
output it stays within `0.011887311935424805` of either nondeterministic
official run; exact equality is neither possible nor meaningful because those
official runs do not equal each other. With this control, repeated B=1
inference is bitwise exact in both stock and invariant-operator modes.

## Baseline and repaired result

The controlled stock path fails every duplicate and unrelated composition
above B=1 across B=`2,3,4,5,7,8,9,15,16,17,31,32,33,64`. At B=2, 2,748 of
2,750 action values differ in both composition modes, with maximum absolute
difference `0.0036270618438720703`.

With invariant operators enabled, the complete output is bitwise exact at
B=`1,2,3,4,5,7,8,9,15,16,17,31,32,33,64` for both compositions. All sizes fit,
so the maximum tested batch is 64. Independent target frames 0, 20, and 40
fail all 12 stock B=2/B=4 checks (maximum difference
`0.015909433364868164`); all 12 repaired checks are exact.

## First divergence and required operators

The real image tensors, Qwen3-VL patch Conv3d, all captured vision blocks, and
all captured language-prefix blocks are exact between B=1 and B=2. The first
divergence is the first flow step's state projection:

```text
input:       [B, 55] float32 (exact)
weight:      [768, 55]
output:      [B, 768] float32
operator:    aten::addmm
B=2 target: 768 / 768 values differ
max error:   0.0004502236843109131
```

The mismatch then propagates through every action-expert block and all later
flow steps. Isolated operator staging gives:

```text
invariant addmm only:       FAIL (2,749 / 2,750 final values)
invariant mm only:          FAIL (2,750 / 2,750 final values)
invariant addmm + mm:       FAIL (2,736 / 2,750, max 3.287e-4)
invariant addmm + mm + bmm: PASS (bitwise exact)
```

`addmm` covers biased state/action projections, `mm` covers bias-free MoE and
MLP projections, and `bmm` covers the official eager-attention einsums. No
model-specific attention replacement is needed for this checkpoint. With all
three reusable kernels, every captured intermediate and final action is exact.

## Performance

At B=1/B=2/B=8, respectively, invariant state-projection addmm is
4.46x/4.18x/4.67x slower than stock, action-expert MM is
4.49x/3.02x/1.15x slower, and action-attention BMM is
2.52x/3.22x/2.67x slower. The dense deterministic MoE fallback is
8.22x/8.31x/14.04x slower than the released nondeterministic atomic kernel.
These are isolated measurements after 50 warmups and across 1,000 repetitions;
complete latency, throughput, shape, and incremental-memory records are in
`benchmarks.json`.

## Evidence boundary

This proves numerical batch invariance for the pinned official PyTorch model,
official checkpoint, FP32 inference path, real three-camera inputs, explicit
per-frame flow noise, all ten sampler steps, and complete 50x55 normalized
output through B=64. It does not claim task success, distribution-matched
RoboTwin evaluation, closed-loop behavior, training invariance, other
checkpoints, or equivalence to any single nondeterministic atomic-MoE run.
