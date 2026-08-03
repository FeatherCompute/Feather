# NN training contract

The host-drivable training surface: a long-lived host process runs a training loop, streams loss out,
checkpoints atomically, and hands trained weights to rendering passes for inference. This is the P0 slice
of [nn-platform-integration.md](nn-platform-integration.md); read that for the full plan and the maturity
table.

## Gradient validation (the gate)

Everything here rests on one thing: the packed-weight MLP training kernel's analytic gradients have to be
correct. EasyGPU documents variable buffer indexing of an AD parameter as non-differentiable, and the
boundary between "index built from loop counters and uniforms" and "index loaded from data" was not
documented precisely.

Measured on `hiddenSize=4`, `sampleCount=6`, `epsilon=1e-2`, all 41 packed weights, against central finite
differences:

| index | analytic | finite difference |
| --- | --- | --- |
| 0 | -0.23004186 | -0.23007393 |
| 1 | -1.166839 | -1.1668563 |
| 2 | 1.4058036 | 1.4057875 |
| 3 | 0.38555875 | 0.38551092 |
| 4 | 1.9323614 | 1.9323587 |
| 5 | -2.3258104 | -2.325797 |

Worst relative error 0.51% across every weight, on `DispatchPath.TypedEasyGpu`.

**Where the boundary actually falls**: an AD parameter read at an index computed from loop counters and
`Uniform<T>` values is differentiated correctly — `weights[layer2Weight + (j * hiddenSize.Value) + k]`
works. The documented limitation applies to *data-dependent* indices, meaning an index loaded from a
buffer. Keep layout arithmetic out of buffers and the packed-weight idiom is safe.

`MlpTrainingGradientTests` pins this.

## The contract

`ITrainingJob` is what a host drives. The host owns the step counter and the reporting cadence; the job
owns device state.

```csharp
var context = new TrainingContext(projectRoot, cancellationToken, lossStream: OnLoss);
using var job = new MlpRegressionJob(hiddenSize, inputs, targets) { PlannedSteps = 600 };
job.Initialize(context);

var last = TrainingStepReport.Unreported(0, DispatchPath.None);
for (var step = 0; step < job.PlannedSteps; step++)
{
    if (context.CancellationToken.IsCancellationRequested) break;
    context.AdvanceTo(step);

    // Only reporting steps pay the readback.
    var report = step % 50 == 0 ? job.StepAndReadLoss(context) : job.Step(context);
    if (report.HasDiverged) break;
    if (report.IsReported)
    {
        last = report;
        context.ReportLoss(report);
    }
}

Checkpoint.SaveAtomic(context.ResolveProjectPath("weights.fthc"), job.Parameters,
    new CheckpointMetadata(last.Step, last.Loss, ModelKind: "mlp-regression-3to1"));
```

`TrainingContext.AdvanceTo` is monotonic and throws on a rewind, so a host bug that replays steps is a
failure rather than silently corrupted provenance. `ResolveProjectPath` rejects rooted and escaping paths,
because a checkpoint path is graph-node data and therefore untrusted.

### Loss readback

`TrainingStep.Run()` reads the loss buffer back every step, which stalls. `RunWithoutLossReadback()` plus
`ReadLoss()` splits that, so a host reporting every 25 steps pays one readback instead of 25. `Run()`'s
behavior is unchanged for existing callers.

A step whose loss was not read reports `Loss` as NaN; check `TrainingStepReport.IsReported` rather than
comparing against zero, which is a legitimate loss.

`Optimizer.Step<TKernel>(GpuADKernel<TKernel>)` is now public. It is the device-only gradient handoff —
gradients never round-trip through the host — and it is what a project with more than one AD kernel per
step needs. `Backward` must have run first.

## Checkpoints

Two on-disk versions. Version 1 is magic, version, count, entries. Version 2 inserts a metadata block
after the version field. Both read. `Save` still writes version 1, so a file written by this build stays
readable by an older one; `SaveAtomic` writes version 2 when metadata is supplied.

| API | Purpose |
| --- | --- |
| `SaveAtomic(path, parameters, metadata?)` | Temp sibling then atomic replace. A crash leaves the previous checkpoint intact. |
| `LoadStrict(path, parameters)` | Throws on a bad file; *reports* name mismatches via `CheckpointLoadResult`. |
| `Inspect(path)` | Header and shapes with no GPU allocation, for a checkpoint picker. |
| `TryReadStamp(path)` | One file stat, for cache invalidation. |
| `Save` / `Load` | Unchanged, including `Load`'s silent-skip semantics. |

Every unreadable-file failure is an `InvalidDataException` — wrong magic, unknown version, truncation,
trailing bytes alike — so a host needs one catch clause. A name mismatch is reported rather than thrown,
because loading a subset of a larger checkpoint is legitimate; call `EnsureComplete()` when it is not.

The temp file is a sibling rather than in the system temp directory, because a move across volumes is a
copy and loses atomicity.

## Inference

`InferenceWeights.Load` reads a checkpoint into GPU buffers once. `Buffer(name)` hands a shader a view —
not a copy, valid only while the instance lives.

`InferenceWeightsCache` is where the lifetime has to sit. A host creates a fresh pass instance per
execution, so a pass cannot hold weights across iterations no matter how it is written; the cache belongs
one level up, alongside the texture and raster-target pools. `GetOrLoad` reloads only when the file stamp
moves, so retraining is picked up on the next iteration with no host restart. The cache owns what it hands
out — callers must not dispose it, and must not call `Load` in a per-frame path.

### The `[Callable]` buffer subset

Typed compute callables can read `ReadOnlyBuffer<T>` parameters for supported scalar and vector element
types. Feather specializes those reads to the bound global SSBO because GLSL cannot pass a runtime-sized
SSBO array as an ordinary function parameter. Writable buffers, buffers in generic interface callables,
GPU-struct buffer elements, textures, and samplers remain outside this callable resource subset.

Consequences:

- `MlpShader` still exposes only layout arithmetic — packed size, scratch stride, per-layer offsets.
  Consolidating evaluation behind a shared callable is separate API work rather than a lowering blocker.
- Batch inference goes through `MlpInference3To1Kernel`, an ordinary kernel that indexes its buffers
  directly.
- The forward arithmetic is therefore duplicated three times: `MlpLayout.Evaluate3To1` (host),
  `MlpRegression3To1LossKernel.Execute` (training), `MlpInference3To1Kernel.Execute` (inference). Nothing
  but tests keeps them in agreement — the gradient tests pin the first two, `MlpInferenceSmokeTests` pins
  the third. The read-only buffer subset now makes a shared `[Callable]` possible, but that consolidation
  is outside this phase-1 lowering change.

## Packed weight layout

A 3→h→h→1 network lives in one flat buffer: `w1[h,3]`, `b1[h]`, `w2[h,h]`, `b2[h]`, `w3[1,h]`, `b3[1]`,
row-major per layer. Two reasons: one `AD.Parameter` marker covers a whole bound buffer, so the entire
network is one parameter group, and an inference kernel's binding count stays at one regardless of depth.

Scratch is `4 * hiddenSize` per lane, sized for training (which keeps pre-activations for the adjoint
pass). Inference needs `3 + 2 * hiddenSize`, which fits for any hidden size of 2 or more — hence
`MlpLayout.MinimumHiddenSize = 2`, so one scratch buffer serves both.

## Sample

`samples/AdMlpRegression` trains on GPU, prints a loss curve, checkpoints atomically, inspects the file,
then loads the weights back and dispatches the inference kernel. 600 steps on 96 samples reduces loss
1.226 → 0.0015.
