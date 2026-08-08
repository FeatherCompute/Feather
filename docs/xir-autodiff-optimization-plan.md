# XIR Autodiff Optimization Plan

## Status and decision

This is a design document for the LuisaCompute XIR reverse-mode autodiff pass.
It replaces the earlier reverse-loop sketch with a plan that covers the whole
expansion pipeline: loop lowering, activity analysis, gradient state, adjoint
generation, memory semantics, and post-AD cleanup.

The measured regression is not explained by push-constant specialization alone.
Specialization is already present in the Feather native path. The dominant
problem is that the current pass materializes a large reverse graph and then
tries to clean it up after the graph has already grown. The recommended order
is:

1. Instrument the pass and make low-risk, activity-aware/compact adjoint
   improvements (P0).
2. Add a restricted reverse-loop transform with recomputation/checkpointing and
   retain bounded unrolling as a correctness fallback (P1).
3. Add memory-aware sharing and fused training patterns after the loop/tape
   contract is stable (P1 follow-up).

The final performance gate is a cold first Metal compile of the 3->12->12->1
MLP in at most 120 seconds and the small GPT demo in at most 300 seconds,
without a numerical regression. Repeated dispatches with the same shader key
should be cache hits and complete in seconds.

## 1. Problem and evidence

### 1.1 Current measurements

All timings below are cold-process measurements on the Feather Metal path unless
otherwise noted. They are evidence for prioritization, not promises about a
particular machine.

| Kernel | Forward/AD shape | Result |
| --- | --- | --- |
| Small linear-regression AD smoke | one simple differentiable expression | about 2.5 s total |
| Literal 12 x 12 AD smoke | two literal-bounded loops | about 0.70 s total, kernel about 254 ms, loss 0.144 |
| 3->12->12->1 MLP, push-constant dimensions | full training kernel | 548.73 s before the first loss; manually stopped |
| Same MLP with literal loop bounds | temporary fixed-dimension kernel | 394.23 s before the first loss; manually stopped |
| MLP-like fixed graph smoke | no dynamic loop cloning | still about four minutes |

The literal MLP result is the key control experiment: removing dynamic loop
cloning does not restore the old one-minute behavior. The large reverse graph,
not only the dynamic-loop gate blocks, is the primary bottleneck.

### 1.2 XIR size growth

Native instrumentation recorded approximately 3,102 instructions before XIR
autodiff and 247,498 after autodiff for the small MLP. That is about 79.8x
growth. The current cleanup sequence removes about 19 percent, leaving roughly
200,000 instructions. Metal compilation is super-linear in this range, so a
19-percent instruction reduction is not a solution.

The dynamic-loop upper bound is still a serious multiplier. The pass defines
`max_ad_loop_unroll_count = 64` and clones a dynamic body up to 64 times. Two
nested dynamic loops can therefore create up to 64 x 64 = 4,096 body instances,
in addition to gate, condition, inactive, and merge blocks. The exact number
executed for a given kernel depends on its structured CFG and trip values.

### 1.3 Specialization and cache semantics already in Feather

The proposed optimization must not be framed as a new "inline Uniform" feature:
the relevant behavior is already implemented.

- In `native/feather_c_api.cpp`, AD lowering sets
  `lowering.dynamic_push_constants = false` and uses the bound push-constant
  bytes while lowering the AD shader (around lines 3300-3312).
- In `native/feather_luisa_xir.cpp`, non-dynamic push constants become XIR
  constants, and integer expressions can be folded from those values (around
  lines 1566-1580 and 2207-2221).
- The AD shader cache key mixes every push-constant binding, size, and byte
  (around lines 3453-3473 of `native/feather_c_api.cpp`). A new dimension value
  therefore gets a distinct shader key; reusing a compiled shader with stale
  dimensions is not the intended behavior.
- Non-AD dispatches deliberately keep push constants dynamic so the ordinary
  kernel cache can be shared safely. This distinction must remain intact.

The performance plan therefore targets the reverse graph and loop transform,
not a cache-key rewrite. Any future specialization change must preserve the
same key rule and must have a test that dispatches the same kernel with two
different dimension values.

## 2. How the current pass expands a kernel

The line references below refer to `LuisaCompute/src/xir/passes/autodiff.cpp`.

### 2.1 Loop analysis and cloning

`analyze_simple_counted_loop` (lines 335-450) proves a fixed trip count only for
a deliberately narrow canonical loop: a local integer alloca, a constant bound,
a constant non-zero step, a prepare block whose non-terminator instructions are
side-effect-free arithmetic/induction loads, and an update branch back to
prepare. It also checks that the induction storage cannot escape through an
alias. A loop that does not satisfy this syntactic proof is not considered
fixed, even when another analysis could prove it.

`unroll_fixed_trip_loop` (lines 853-962) clones the body region once per proven
trip. It clones prepare instructions into the body clone, snapshots values that
escape the loop, lowers PHIs to local storage, retargets backedges, and emits a
final prepare evaluation before the merge. This is semantically careful but
makes code size proportional to the trip count.

`unroll_bounded_dynamic_loop` (lines 981-1169) handles a canonical dynamic loop
by allocating a done flag and six control blocks per possible iteration. Each
iteration clones the complete loop region, evaluates the original prepare
condition, and gates the clone. An overflow check emits `unreachable` when the
64-iteration budget is exceeded. The implementation is correct for its stated
bounded contract, but nested use multiplies clone count.

`unroll_fixed_trip_loops` (lines 1224-1238) repeatedly finds first-level loops
and applies one of those two clone paths. There is currently no third path for
a reverse loop; `collect_forward` and `emit_backward_instructions` reject
remaining loop instructions.

### 2.2 Forward activity and state allocation

`TransformAdScope` (lines 462 onward) tracks forward-reachable values,
backward-reachable values, forward instruction order, branch snapshots, switch
snapshots, and removable AD intrinsics.

- `grad_slot` (lines 1320-1339) creates a function-wide alloca for each
  differentiable value that becomes relevant and inserts a zero store at the
  scope boundary. The alloca lifetime is function-wide, while the logical
  gradient lifetime is one dynamic execution of the scope.
- `process_forward_instruction` (lines 1488-1577) propagates activity through
  GEP, load/store, arithmetic, cast, and call instructions. This is a useful
  starting point for a real activity-aware tape, but it does not yet group
  values by loop iteration or operation pattern.
- `collect_forward` (lines 1579-1649) walks the structured CFG, records every
  instruction up to `backward()`, and saves branch/switch conditions for the
  reverse control flow. Loops must already have been cloned away.

The pass does not have a compact tape abstraction. Reverse code refers to the
forward instruction values and uses gradient allocas plus branch/switch
snapshots as mutable state. In a generated training kernel this produces many
loads, additions, and stores around the actual arithmetic. Scratch-buffer
values are also conservatively treated as mutable memory by downstream CSE.

### 2.3 Reverse generation and gradient accumulation

`collect_backward` (lines 1651-1688) scans the recorded forward instructions in
reverse order to discover which operands need gradients. `emit_backward` and
`emit_backward_instructions` (lines 2312-2337) then revisit each instruction in
reverse order.

`backward_arithmetic` (lines 1785-2267) has a hand-written case for each
arithmetic operation. A single forward operation can emit several new XIR
operations for its adjoint. For example, multiplication emits one product for
each active operand; division emits a square, a negated factor, and two
products; matrix multiplication emits transposes and matrix products. The
formulas are generated independently at each occurrence, before graph-level
sharing is attempted. There is even an obviously dead temporary in the DOT
case (`lhs_mul_rhs`, around lines 2108-2112), illustrating why a compact
adjoint IR should avoid materializing throw-away intermediates in the first
place.

`accumulate_grad` and `accumulate_into_lvalue` (lines 1387-1459) implement
gradient accumulation as load + add + store into an alloca. GEP gradients are
rebuilt with aggregate INSERT/EXTRACT operations. This is a safe generic
contract, but it prevents ordinary value numbering from treating two loads as
the same value when an intervening store or alias may exist.

Store and load instructions are handled by `backward_inst` (lines 1759-1782):
stores propagate the lvalue gradient to the stored value and then clear the
lvalue gradient. That ordering is required for mutable locals, and any memory
optimization must preserve it.

Branch conditions are snapshotted by `snapshot_if_condition` and
`snapshot_switch_value` (lines 1352-1372), then replayed by
`emit_backward_if`/`emit_backward_switch` (lines 2270-2310). A condition inside
a loop needs a per-iteration value or a provably equivalent recomputation;
one function-wide slot is not sufficient for a reverse loop.

### 2.4 Surrounding pass pipeline

The native AD path in `native/feather_luisa_backend.cpp` currently performs:

1. Destructure and inline the callable graph.
2. GVN, DCE, CFG simplification, and DCE before restructuring.
3. Destructure/reg2mem/restructure, then `autodiff_pass_run_on_module`.
4. Verify, destructure/reg2mem/restructure again.
5. GVN, DCE, CFG simplification, and DCE after AD (around lines 2483-2553).

The XIR GVN pass intentionally value-numbers arithmetic, casts, GEPs, and safe
resource queries only. It excludes ordinary loads and resource reads because it
has no memory-dependency/alias proof (`src/xir/passes/gvn.cpp`, lines 119-141).
It also avoids reordering floating-point add/mul unless the type is integral
(lines 35-57). Those are correct defaults, but they explain why cloned scratch
loads and strict-floating-point arithmetic remain distinct. A broad temporary
cleanup experiment (constant/algebraic simplification, SCCP, early CSE, local
load/store elimination, and dead-store elimination) did not improve the cold
Metal compile and raised compiler memory to about 1.3 GB; that experiment was
reverted. Cleanup should be added only with pass statistics and verifier tests.

## 3. Where the expansion comes from

The following accounting separates measured facts from structural estimates.

| Source of growth | Multiplier or effect | Evidence and implication |
| --- | --- | --- |
| Dynamic loop cloning | Up to 64 clones per dynamic loop; nested loops multiply (up to 4,096 for two levels) | `unroll_bounded_dynamic_loop`; the gate/eval/merge scaffolding is also cloned |
| Fixed loop cloning | One complete body clone per trip | Literal MLP still took 394.23 s, so this is not the only bottleneck |
| Per-instruction adjoint | One reverse visit for every active forward instruction; each arithmetic case emits multiple operations | `collect_backward` plus `backward_arithmetic`; 3,102 -> 247,498 measured |
| Gradient state | A function-wide alloca and load/add/store sequence per active value and accumulation | `grad_slot`, `accumulate_grad`, `accumulate_into_lvalue` |
| Mutable memory | Loads/stores from scratch and gradient slots cannot be CSE'd without alias epochs | GVN explicitly excludes LOAD and RESOURCE_READ |
| Branch/tape state | Condition/switch snapshots and reverse control blocks | `snapshot_if_condition`, `snapshot_switch_value`, `emit_backward_if/switch` |
| Late cleanup | Only arithmetic/cast/GEP sharing is available by default; memory is conservative | Existing post-AD cleanup removes about 19 percent, leaving about 200k instructions |

The MLP source has an outer hidden-neuron loop with a 3-wide dot product, an
outer hidden-neuron loop with a hidden-by-hidden dot product, and an output
dot product (`src/Feather/NN/MlpTraining.cs`, lines 198-230). The hidden-by-
hidden product is the largest repeated arithmetic region. Every multiply/add,
ReLU comparison, scratch load, and scratch store that is active in the loss
path gets a reverse visit; its gradients then flow through mutable scratch
allocations. This accounts for the large graph even when H is a literal.

The GPT kernel contains many more nested products: position x embedding
initialization, normalization reductions, query/key/value projections,
position x head x time x head-size attention, softmax reductions, projection,
MLP, and vocabulary logits (`src/Feather/NN/SequenceModels.cs`, lines
1136-1349). Its structure is an especially strong candidate for reverse loops,
checkpointing, and fused reduction rules.

### Required temporary instrumentation

Before implementing a large transform, add instrumentation locally (do not
commit it) or expose equivalent `PassReport` counters. At minimum record:

- instruction counts by `DerivedInstructionTag` before loop lowering, after
  loop lowering, after forward collection, after backward emission, and after
  each cleanup pass;
- fixed-loop trip counts, dynamic-loop clone counts, and the number of cloned
  body instructions per nesting level;
- number and total byte size of gradient slots, branch/switch snapshots, and
  any future tape/checkpoint slots;
- counts of arithmetic adjoint cases and generated load/add/store/INSERT/
  EXTRACT instructions;
- number of GVN candidates rejected specifically because they are LOAD or
  RESOURCE_READ, plus memory-write invalidation epochs if a memory-aware GVN is
  prototyped;
- XIR-to-AST time and backend Metal compile time separately.

The resulting histogram should be saved with the benchmark artifact, not in
the source tree. It is the gate for deciding whether a proposed optimization
actually removes graph size rather than moving time between passes.

## 4. Optimization directions

### Comparison

| Direction | Expected code-size/compile benefit | Complexity | Main risk | Scope |
| --- | --- | --- | --- | --- |
| A. Reverse-graph CSE and sharing | High for repeated dot/GEMM chains; 2-10x on suitable graphs | High | Unsound memory CSE or changed FP behavior | General XIR |
| B. Compact adjoint generation | Medium; often 1.5-3x and lower peak IR memory | Medium | Missing a contribution or changing accumulation order | General XIR |
| C. Reverse loops | Very high; removes trip-count multiplier, potentially 10-100x | Very high | Wrong loop-carried state, early exits, or tape values | General canonical loops |
| D. Value cache/checkpointing | High when scratch values dominate; trades memory for recompute | High | Aliasing and numerical differences during recompute | General XIR with restrictions |
| E. Post-AD GVN/DCE/algebraic cleanup | Low to medium; current baseline is about 19 percent | Low to medium | Memory blow-up and compile time spent in cleanup | General XIR |
| F. Training-pattern adjoints | Very high for GEMM/ReLU/softmax kernels | High | Pattern brittleness and domain-specific semantics | NN kernels |

### A. Reverse-graph common-subexpression sharing

**Principle.** Build an adjoint DAG or perform graph-level value numbering
before materializing each gradient store. Contributions that use the same
weight, activation, or reduction term should share the common value. A matrix
product should expose a single transposed/shared operand and a batched
contribution, rather than independently rebuilding the same chain for every
neuron.

**Implementation.** Start with SSA arithmetic/cast/GEP values and immutable
resource reads. Add a memory version/alias epoch to the expression key. A
read-only buffer read is shareable when resource identity and all indices are
identical and no write to the resource's alias class intervenes. A write, an
atomic, a volatile read, an unknown call, or an escaped pointer increments or
invalidates the epoch. Do not make strict floating-point add/mul commutative;
only reassociate under an explicit fast-math option.

**Benefit and cost.** This can remove most repeated arithmetic in a dense
backward graph and lowers both XIR size and Metal front-end work. It requires a
memory-dependency analysis stronger than current GVN and a cost model to avoid
increasing register pressure. Complexity is high; the first version should be
limited to local SSA and proven read-only resources.

**Risk/impact.** The main failure mode is a stale load after a scratch write.
Keep the existing conservative fallback for unknown memory and add litmus
tests with aliasing, volatile operations, NaNs, signed zero, and multiple
gradient consumers.

### B. Compact reverse/adjoint generation

**Principle.** Treat reverse generation as construction of a small gradient
graph, not as an immediate instruction-by-instruction append. Delay materializing
`load(slot)`, `add`, and `store(slot)` until all contributions in a basic block
are known. Skip inactive operands early and remove adjoint temporaries whose
result is unused.

**Implementation.** Replace the eager accumulation helper with a per-scope
pending-contribution map. Combine contributions in source order, then emit one
store per slot at a safe boundary. Add operation-specific compact rules for
FMA, multiply/add chains, select/ReLU masks, and reductions. Reuse a forward
value when its lifetime and dominance are proven; otherwise create a single
checkpoint or recompute it. Preserve the current operation order by default;
fused or reassociated forms require fast-math or a documented numerical mode.

**Benefit and cost.** This reduces alloca traffic and peak builder memory even
before reverse loops exist. It is medium complexity and should be the first
code-generating optimization after instrumentation. It cannot remove the
64 x 64 trip multiplier by itself.

**Risk/impact.** Incorrectly coalescing two stores changes mutable-local
semantics. Emit at block boundaries where dominance is known and retain the
existing load/add/store path for cross-block or alias-sensitive values.

### C. Reverse loop instead of static cloning

**Principle.** The adjoint of a counted loop is a loop that visits iterations in
reverse order. Its code size is proportional to the loop body, not the runtime
trip count. Nested loops reverse inside-out.

**Initial eligibility.** Match only canonical `LoopInst` counted loops after
`reg2mem`/`restructure`: a single induction alloca, a monotonic constant step,
one prepare condition, one update edge, no nested loop in the region, and no
`break`/`continue` that changes trip count. Reject `SimpleLoopInst`, unknown
calls, atomics, volatile accesses, and irreducible CFGs. Loops that fail the
proof continue through bounded unrolling; `max_ad_loop_unroll_count` remains a
fallback and is not reduced.

**Reverse shape.** Compute the executed trip count (or preserve a runtime
counter) and create a reverse `LoopInst` with an induction value initialized to
the last executed iteration. Decrement before the reverse body, emit the
existing backward instruction rules for the body, then branch to the reverse
prepare. The reverse condition must use the same signedness, inclusive/exclusive
comparison, overflow behavior, and loop-invariant values as the forward loop.

**Required state.** A reverse loop must recover values that the forward body
used. The first implementation should support:

- recomputation of pure arithmetic from loop-invariant inputs and the reverse
  induction value;
- persistent scratch values whose alias analysis proves that a later forward
  iteration cannot overwrite the value needed by an earlier reverse iteration;
- per-iteration snapshots for branch predicates and loop-carried scalars when
  recomputation is not proven equivalent.

Loops with arbitrary mutable scratch aliases or early exits fall back to
bounded cloning until a tape implementation supports them. This restriction is
important: simply emitting a reverse loop while reading final scratch contents
would silently compute the wrong gradient.

**Benefit and cost.** This is the root fix for the trip-count explosion and is
expected to bring the MLP into the two-minute target once compact state is in
place. It is very high complexity because it changes CFG ownership and the
value-recovery contract of `TransformAdScope`.

### D. Value cache, activity analysis, and checkpointing

**Principle.** Cache only values needed by the reverse graph. Rematerialize
cheap pure expressions and checkpoint expensive or mutable values. Treat a
gradient slot, a forward value, a branch predicate, and a loop-carried value as
different classes of state; they should not all be allocated with the same
function-wide lifetime.

**Implementation phases.**

1. Extend the existing forward/backward reachability to an activity lattice:
   active value, active memory location, active predicate, loop-carried, and
   dead. Do not allocate `grad_slot` until a value has a backward consumer.
2. Build a per-region tape plan. For a pure expression, store its operands or
   recompute it. For a mutable load, store a checkpoint only if alias analysis
   cannot prove stable contents. For a loop, choose full per-iteration tape,
   periodic checkpoints plus recomputation, or bounded fallback.
3. Lower tape entries to an explicit XIR scratch buffer or a verified local
   array representation. Dynamic alloca arrays and growable stacks are out of
   scope for the first milestone.

**Benefit and cost.** This controls both device memory traffic and host compiler
peak memory. It is high complexity and should be implemented together with the
reverse-loop plan, because a reverse loop without a tape contract is not
correct.

**Risk/impact.** Recompute can change floating-point results if it changes
evaluation order or observes a modified resource. Preserve the original order,
avoid reassociation, and use finite-difference/analytic gradient tests with
strict tolerances. Expose a diagnostic when a loop is rejected for unsupported
state rather than silently selecting an unsafe recovery strategy.

### E. Expanded post-AD cleanup

**Principle.** Run cleanup while the graph is still small and add narrowly
scoped memory-aware passes after AD. Existing `gvn`/`dce`/`simplify_cfg` are
useful but cannot identify equal loads through mutable scratch.

**Implementation.** Use the pass pipeline APIs for pass reports and verifier
checkpoints. Add, in order, const-fold/algebraic simplification, early CSE for
SSA-only values, local store/load forwarding, dead-store elimination, then
GVN/DCE. Add memory-aware CSE only after alias epochs are available. Keep a
strict instruction-count and wall-time budget for every cleanup pass.

**Benefit and cost.** Low to medium complexity and useful as P0 hygiene. It
cannot solve a 200k-instruction reverse graph; the prior broad cleanup trial
showed no compile-time improvement and about 1.3 GB compiler memory. Do not
claim E alone meets the final target.

### F. Training-kernel pattern matching

**Principle.** Recognize stable NN idioms and emit one compact adjoint pattern:

- GEMM/dot-product: share the reduction and use transpose products for weight
  and activation gradients;
- ReLU/max: retain or recompute one comparison mask and apply a select;
- softmax/logsumexp: share max, denominator, and normalized probability terms;
- layer normalization: share mean/variance/inverse-scale terms;
- optimizer/update kernels: keep them outside the tape when they are not part
  of the differentiated loss.

**Implementation.** Prefer an XIR-level reduction/matrix idiom matcher or
explicit intrinsic patterns over matching C# syntax. Match only canonical,
side-effect-free regions and fall back to generic adjoints otherwise. Consider
introducing a fused GEMM backward intrinsic only after the backend contracts,
buffer layouts, and gradient accumulation order are specified.

**Benefit and cost.** Potentially the largest domain-specific gain for GPT,
but high implementation and maintenance cost. It increases the impact surface
to NN kernels and needs per-pattern numerical tests, including degenerate
dimensions and masks.

## 5. Industry references and applicable ideas

| System | Relevant strategy | What XIR should reuse | What not to copy directly |
| --- | --- | --- | --- |
| Tapenade | Activity analysis, source-level reverse transformation, loop reversal, checkpointing | Explicit active-variable/tape planning and checkpoint placement | Source-language assumptions about aliasing and structured loops |
| Enzyme | LLVM SSA activity analysis, augmented returns, alias-aware memory handling, reverse loops | Treat memory effects and activity as first-class analyses; preserve an explicit fallback for unknown aliasing | LLVM-specific intrinsics and unrestricted pointer assumptions |
| JAX/XLA | Primitive transpose rules, static shapes, HLO fusion, rematerialization/checkpoint policies | Operation-level VJP rules, fused reductions, shape-aware cost model | Python tracing/runtime polymorphism; XIR must keep a verified low-level fallback |
| PyTorch AOTAutograd | Functionalized forward graph, operator VJP graph, compiler fusion and partitioning | Separate forward graph from VJP graph, partition around mutable state, fuse stable operator regions | Treating arbitrary in-place mutation as functional without an XIR memory proof |

The common lesson is to construct a compact reverse graph using activity and
memory analyses before lowering to target code. All four systems retain a
fallback for operations or control flow they cannot prove safe; XIR should do
the same instead of guessing a reverse-loop tape.

## 6. Recommended implementation route

### P0: measurement and low-risk reduction

P0 is intended to reduce peak IR and establish hard data. It is not expected to
remove the fundamental trip-count multiplier by itself.

#### P0.1 Instrumentation and baseline

- Add temporary counters described in section 3 and record cold/warm timings
  for MLP and GPT with fixed push-constant values.
- Verify the existing AD specialization/cache behavior by dispatching two
  dimension values and checking that they produce two shader keys and correct
  results.
- Add verifier checkpoints after inlining, before AD, after AD, after cleanup,
  and before XIR2AST.

Exit criterion: a reproducible report containing instruction counts, clone
counts, slot/tape bytes, pass times, XIR2AST time, Metal compile time, and
cache-hit time.

#### P0.2 Compact adjoint and lazy state

- Do not allocate a gradient slot until backward reachability proves a use.
- Build pending gradient contributions per basic block and emit one safe
  accumulation sequence where possible.
- Remove trivially dead adjoint temporaries before inserting them into the
  module; keep source-order accumulation under strict floating-point mode.
- Add compact rules for FMA, multiply/add chains, ReLU/select, and reductions.

Exit criterion: MLP post-AD instruction count and peak pass memory decrease
without a correctness failure; no dimension or control-flow case may produce a
larger graph than the current fallback. P0 should beat the 394-548 second
baseline, but the two-minute/300-second goals are P1 gates.

#### P0.3 Safe cleanup and local sharing

- Run cleanup before AD whenever it does not alter autodiff scope structure.
- Run SSA-only early CSE and local load/store forwarding after AD.
- Prototype memory-aware CSE only for immutable resources and proven local
  allocas. Keep unknown resource reads and writes conservative.
- Record every pass's removed instruction count and wall time; reject a pass if
  it increases peak memory or total cold compile time on the benchmark matrix.

### P1: remove the graph-size root cause

#### P1.1 Activity-aware tape

Implement the activity lattice and tape plan from direction D. Initially support
pure counted loops and stable scratch reads, with explicit rejection for
unknown aliases, atomics, volatile operations, calls, and early exits.

#### P1.2 Canonical reverse loops

Add `ReverseLoopPlan` selection beside
`unroll_fixed_trip_loop`/`unroll_bounded_dynamic_loop`:

1. Analyze induction variable, bound, step, prepare purity, exits, and nested
   ownership.
2. Analyze active values and memory writes in the loop body.
3. Select recomputation, checkpoint, or bounded fallback.
4. Emit the reverse loop and per-iteration predicate/loop-carried state.
5. Run verifier and a numerical gradient check. On any unsupported shape,
   leave the original bounded lowering path intact.

Do not change or lower `max_ad_loop_unroll_count` as a performance workaround.
The bounded path is needed for semantic coverage while reverse-loop support is
incrementally expanded.

#### P1.3 Fused NN patterns

After the generic reverse loop and tape contracts are stable, add XIR-level
patterns for GEMM/dot, ReLU, softmax/logsumexp, and layer normalization. Each
pattern must have a generic fallback and an independent numerical test.

## 7. Reverse-loop implementation details

### 7.1 IR contract

For a forward loop with `i = start`, condition `i < bound`, and step `step`,
the reverse scaffold is conceptually:

```
trip = executed_trip_count
i = trip
reverse_loop {
    if (i == 0) break
    i = i - 1
    recover_forward_values(i)
    emit_adjoint_of_body(i)
}
```

The actual XIR representation must use structured `LoopInst` ownership and
preserve metadata. Inclusive comparisons, negative steps, integer wraparound,
and a zero-trip loop must be handled explicitly. Trip-count arithmetic must not
overflow differently from the forward loop.

### 7.2 State and mutation

- Initialize loop-carried gradient slots once before entering the reverse loop;
  do not clear them on every iteration.
- Reverse stores in the same order as the source's reverse instruction order,
  then clear the gradient of the overwritten lvalue.
- Save a branch predicate per iteration or recompute it from a value whose
  version is proven stable. A single scope-wide condition slot is insufficient.
- For nested loops, establish the outer iteration state before entering the
  inner reverse loop and restore it after the inner loop completes.

### 7.3 Recompute/checkpoint policy

The first implementation should be conservative and bounded:

- Pure arithmetic and loop-invariant reads: recompute from `i` and invariant
  operands.
- Values written to a non-aliased scratch range and never overwritten before
  reverse use: reload from scratch.
- Mutable or aliased values: checkpoint per iteration in an explicit tape
  buffer, or select bounded unrolling.
- Early `break`/`continue`, calls, atomics, and volatile accesses: fallback.

Periodic checkpoints can reduce tape memory after correctness is established,
but only with a cost model that compares recomputation work against device
memory traffic. A naive "replay from the start for every reverse iteration"
algorithm is O(n squared) and is not acceptable for GPT attention loops.

## 8. Acceptance and regression plan

### 8.1 Performance gates

Measure cold process compile and warm dispatch separately, using the same
backend, device, dimensions, and shader cache directory.

| Gate | Target |
| --- | --- |
| 3->12->12->1 MLP, first Metal compile | <= 120 s |
| Small AdGptDemo, first Metal compile | <= 300 s |
| Repeated dispatch with identical AD key | seconds-level cache hit |
| XIR instruction count | no more than 2x the compact/reverse-loop budget chosen by the implementation; track absolute count in the report |
| Training throughput after compile | no more than 10 percent regression versus the pre-optimization kernel |

P0 is successful only if it produces a measurable reduction with no regression;
P1 must meet the final compile gates. If a reverse-loop candidate is rejected,
the report must name the rejected proof and show bounded fallback behavior.

### 8.2 Correctness tests

Add LC unit tests for:

- fixed and runtime canonical counted loops with trip counts 0, 1, 2, and the
  maximum supported test value;
- nested reverse loops and unequal trip counts;
- loop-carried scalar gradients and mutable local stores;
- branch predicates inside loops, including ReLU-like masks;
- early exit, continue, atomics, volatile/resource writes, and unknown aliases
  selecting the bounded fallback or a clean unsupported diagnostic;
- strict floating-point cases with NaN, signed zero, and cancellation.

Run the existing XIR AD unit/integration suites and finite-difference checks.
The reverse-loop transform must pass the verifier after every CFG conversion.

### 8.3 Feather regression matrix

For every milestone, run:

- Feather AD tests (66), NN tests (78), and Integration tests (300+), all green;
- `AdMlpRegression` full training with loss decreasing and exit code 0;
- `AdGptDemo` small configuration through all requested training steps,
  recording first compile time, loss samples, and exit code;
- representative `AdLinearRegression`, `AdTransformer`, and
  `AutoDiffLinearRegression` runs;
- at least one non-AD sample and one dimension change to verify that ordinary
  dynamic push constants and cache behavior remain correct.

Record native build/staging versions and the LC commit used for each benchmark.
Do not accept a result that only uses a warm shader cache.

## 9. Risks and explicit non-goals

- Reverse loops are not initially required to handle arbitrary CFG, `break`/
  `continue`, `SimpleLoopInst`, opaque calls, atomics, volatile memory, or
  escaped pointers. Those shapes must use the existing bounded path or a clear
  unsupported error.
- Dynamic growable tape stacks and unbounded local arrays are out of scope for
  the first milestone.
- Parallel execution of reverse iterations is out of scope; data dependencies
  require sequential reverse order unless a separate reduction proof exists.
- Do not reduce `max_ad_loop_unroll_count` to hide compile time. It changes
  program semantics for valid dimensions and only moves the failure to runtime.
- Do not change non-AD push-constant dynamics or omit push-constant bytes from
  an AD shader cache key.
- Do not rely on backend-specific Metal optimizations to compensate for an
  oversized XIR graph.

The decisive architectural boundary is the tape/memory contract. Once XIR can
prove which values are active and how a loop iteration's memory can be
recovered, the reverse-loop transform removes the multiplicative clone factor.
Until then, compact adjoints and conservative memory-aware sharing provide
useful P0 wins without compromising correctness.
