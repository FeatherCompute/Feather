# NN Platform Integration

This document specifies the SDK work required to make Feather's neural-network
story usable *inside the Blender node-graph platform*, not only from standalone
console samples. The requirement is that a project can train a model on the GPU,
stream live loss to the host, checkpoint into the project directory, and then
consume the trained weights from a rendering pass.

It is a design document driven by an audit of the current source tree. Every
"today" claim below cites a file and line. Every proposed API is marked as new.

- Audience: implementers of the Feather SDK and of the Blender fork add-on.
- Non-goal: turning `Feather.NN` into a general ML framework. See
  [Feather.NN Status](nn-status.md) for the deliberate scope.

## (a) Current State

### Maturity Table

`Feather.NN` and `Feather.AD` public types, as they exist today.

| Type | File | Maturity | Notes |
| --- | --- | --- | --- |
| `Tensor<T>` | `src/Feather/NN/Tensor.cs:9` | Preview-stable | Shape + `GpuBuffer<T>`; `AsReadOnlyBuffer()`/`AsReadWriteBuffer()` at lines 72/77 give a shader-facing view. |
| `TensorShape` | `src/Feather/NN/Tensor.cs:96` | Preview-stable | Validates positive dims; value equality. |
| `TensorView<T>`, `Tensor2D<T>` | `src/Feather/NN/TensorViews.cs:9,54` | Preview | `TensorView<T>` carries an `Offset` but **no op consumes it** — no kernel or `NnDeviceOps` entry takes a view. Effectively inert today. |
| `IParameter` / `Parameter<T>` | `src/Feather/NN/Tensor.cs:166,210` | Preview-stable | Value + gradient tensor, `FullName`, `GradientNames`, `AddGradientAlias`. `ZeroGrad()` is float-only (line 261). |
| `ParameterGroup`, `ParameterInitializers` | `src/Feather/NN/ParameterGroups.cs:6,50` | Preview-stable | Xavier/constant init; per-group LR/weight-decay. |
| `Module`, `Linear`, `Embedding`, `LayerNorm`, `BatchNorm1D`, `Sequential` | `src/Feather/NN/Modules.cs:11,40,117,219,296,409` | Preview | Forward paths dispatch real kernels. **Forward is inference-only** — it does not build any graph usable for training. |
| `Activation` + `ReLU`/`Sigmoid`/`Tanh`/`SiLU`/`Softmax`/`LogSoftmax` | `src/Feather/NN/ActivationsAndLosses.cs:6,29–74` | Preview | Device dispatch via `NnActivationKernel`. |
| `TensorOps`, `Losses`, `CrossEntropyLoss` | `src/Feather/NN/ActivationsAndLosses.cs:83,167,353` | Preview | Scalar-returning overloads read back one value by design (`nn.md:108`). |
| `NnDeviceOps`, `NnDispatchTrace` | `src/Feather/NN/DeviceOps.cs:57,24` | Preview | The real device op layer; `NnLinearForwardKernel` at line 400 is forward-only, **not** AD-authored. |
| `Optimizer`, `SGD`, `Adam`, `AdamW`, `RMSProp` | `src/Feather/NN/Modules.cs:504,779,832,1016,1089` | Preview-stable | State in GPU tensors; `Parameter<float>` only (line 515). |
| `TrainingStep<TKernel>` | `src/Feather/NN/Modules.cs:670` | Preview | The supported training contract. `Run()` at line 725. |
| `AD` markers | `src/Feather/AD/AD.cs:15` | Preview-stable | `Parameter(float/float2/3/4)`, `Loss(float)`; non-scalar `Loss` overloads exist only to be rejected in lowering (lines 28–32). |
| `GpuADKernel<TKernel>` | `src/Feather/AD/AD.cs:43` | Preview | `Forward`/`Backward`/`CopyGradientToBuffer`/`ReadBackGradients`/`GetBackwardGLSL`. |
| `GradientSet` | `src/Feather/AD/AD.cs:331` | Preview | Lazy; `ReadBackGradients()` is the debug path. |
| `Checkpoint` | `src/Feather/NN/Checkpoint.cs:6` | Preview, thin | `Save`/`Load` only. `FTHC` magic + version 1 (lines 8–9). Float parameters only (line 20). Non-atomic `File.Create` (line 22). |
| `PositionalEmbedding`, `SelfAttention`, `TransformerBlock`, `GptLanguageModel`, `SelfAttentionClassifier` | `src/Feather/NN/SequenceModels.cs:11,85,206,335,668` | Preview, sample-shaped | Built for the samples, not general. |
| `GptLanguageModelTrainer`, `SelfAttentionClassifierTrainer` | `src/Feather/NN/SequenceModels.cs:509,739` | Preview, sample-shaped | Constructors are `internal`; created via `CreateTrainer` (line 417). |

### The Central Constraint: One Monolithic AD Kernel Per Step

This is the fact that shapes everything below.

Training a multi-layer model today means **fusing the entire forward pass and the
loss into a single `[AutoDiff] IKernel1D` struct**. `GptLanguageModelTrainingKernel`
(`src/Feather/NN/SequenceModels.cs:1077`) takes token embedding, positional
embedding, packed attention weights, `fc1`, `fc2`, and the LM head as six
`ReadWriteBuffer<float>` resources plus one `scratch` workspace buffer, and runs
embedding lookup, LayerNorm, causal attention, residual, MLP, logits, and
cross-entropy in one `Execute()` body (lines 1094–1351). It ends with six
parameter markers and one loss marker (lines 1344–1350).

There is **no chained AD**: no kernel's output buffer feeds a second kernel with
gradients flowing back across the dispatch boundary. `GptLanguageModelTrainer`
holds exactly one `GpuADKernel<...>` field (line 516) and `TrainBatch` is
`Backward` → `optimizer.Step` → loss readback (lines 572–579). The same is true
of `SelfAttentionClassifierTrainer` (lines 745, 797–803).

Two consequences worth stating plainly:

- The `Module`/`Sequential` object model and the AD training path are **disjoint**.
  `Linear.Forward` dispatches `NnLinearForwardKernel`, which carries no AD
  metadata. Training a `Sequential` requires hand-writing a kernel that indexes
  the same parameter buffers — exactly what
  `tests/Feather.Integration.Tests/NNTrainingIntegrationTests.cs` does with the
  per-test `NNSequentialReluMlpMeanLossKernel`.
- The intermediate activations *inside* one AD kernel do cross a buffer
  (`scratch`), and gradients do flow through it, because it is all one adjoint
  body. That is what the AD note means by gradients through intermediate
  buffers. It is **not** gradients across dispatches.

One marker covers a whole buffer. `AD.Parameter(weights[0])` registers the entire
bound buffer as a parameter group: the native bridge resolves the binding, computes
`element_count` from buffer size / stride, and calls
`gradientTape.RegisterBufferParameter(name, glslType, element_count)`
(`native/feather_c_api.cpp:5494–5516`). It rejects anything that is not a
differentiable buffer element (line 5470). So a 4096-element weight matrix costs
one marker, not 4096.

### AD Supported Subset

From `docs/autodiff.md:147` and `docs/ad-implementation-note.md:54`, confirmed
against the native bridge:

- Generated **1D** kernels only (`GpuADKernel<TKernel>` constrains `IKernel1D`,
  `src/Feather/AD/AD.cs:44`).
- One **scalar** loss.
- Structured `if`/`else` and canonical counted `for`. `while`, `do-while`,
  `break`, `continue` rejected.
- `float`, `float2/3/4` parameters, buffer-element sourced.
- EasyGPU adds a constraint the Feather docs do not state:
  **variable buffer indexing of a parameter is not differentiated** —
  `EasyGPU/docs/autodiff.md:786` says `buf[varIndex]` with a non-constant index
  "prevents the system from tracking which parameter is being updated". The
  existing GPT kernel works because parameter reads are inside counted `for`
  loops the adjoint generator can reverse. Any new AD kernel must be validated
  against this, not assumed.

### Host Boundaries Today

- **Loss readback is synchronous when requested.** `TrainingStep.Run()` calls
  `lossBuffer.ToArray()` on every invocation (`src/Feather/NN/Modules.cs:732`),
  as does `GptLanguageModelTrainer.UpdateDiagnosticsFromLossBuffer` (line 635 of
  `SequenceModels.cs`). `GpuBuffer<T>.Read` and `ToArray` are whole-buffer and
  blocking (`src/Feather/Resources/GpuBuffer.cs:135,165`). `RunWithoutLossReadback()`
  skips that readback but preserves synchronous completion; queue-owning hosts can use
  `EnqueueWithoutLossReadback()` and an explicit `GPU.Queue` fence. There is no async or
  ranged buffer readback.
- **No callbacks anywhere.** No `IProgress<T>`, no `CancellationToken`, no events
  in `Feather.NN` or `Feather.AD`. Every sample owns its own `for` loop and its
  own `Console.WriteLine` cadence: every step in `samples/AdLinearRegression`,
  every 40 in `AdTransformer`, every 50 in `AdGptDemo`, every 250 in
  `AdGptPoetDemo`. Divergence handling in `AdGptPoetDemo` is a hand-rolled
  `break`, not an SDK hook.
- **No sample saves a checkpoint.** `Checkpoint.Save`/`Load` appear only in tests
  (`tests/Feather.NN.Tests/NNSurfaceTests.cs:1103,1126,1157` and
  `tests/Feather.Integration.Tests/NNTrainingIntegrationTests.cs:678`). No
  sample writes one, so there is no established on-disk location convention.
- **No CLI surface.** Every NN sample is a top-level-statement console program
  with hardcoded hyperparameters. None parses arguments.
- **Device-side optimizer handoff is public.** `Optimizer.Step<TKernel>(GpuADKernel<TKernel>)`
  supports custom multi-kernel training drivers without a managed gradient readback. The
  non-blocking `TrainingStep` path is available to built-in optimizers; custom optimizers
  must explicitly implement the protected asynchronous submission contract.
- **The RenderHost knows nothing about NN.** Grepping
  `src/Feather.Blender.RenderHost` and `src/Feather.ManifestExporter` for
  `Feather.NN`, `Feather.AD`, `Checkpoint`, `TrainingStep` returns nothing.

### What Already Works For Inference (Better Than Expected)

Three pieces of the inference story exist today and should not be rebuilt:

1. **Weights are already GPU buffers.** `Parameter<float>.Value` is a
   `Tensor<float>`, and `Tensor<T>.AsReadOnlyBuffer()`
   (`src/Feather/NN/Tensor.cs:72`) yields a `ReadOnlyBuffer<float>` that can be
   passed straight into a generated kernel struct. No new interop layer is needed
   to get weights into a kernel.
2. **Fragment shaders can read buffers.** `MinimalRasterFragmentShader` takes a
   `ReadOnlyBuffer<MinimalRasterLight>` and loops over it
   (`samples/BlenderRenderGraph/Passes/MinimalRasterPass.cs:377,422`). So an MLP
   can be evaluated in a raster pass, not just a compute pass.
3. **Float textures exist.** `PixelFormat.R32Float`, `Rg32Float`, `Rgba32Float`
   (`src/Feather/Core/Enums.cs:20–35`) with `AsSampled()` and
   `Sample`/`SampleLevel`/`SampleGrad` (`src/Feather/Resources/GpuTexture2D.cs:140`,
   `501–506`). Hardware-filtered weight or feature-grid lookup is available if a
   design wants it.

The gaps for inference are therefore **not** interop primitives. They are:
path resolution, per-frame reload cost, and the absence of a reusable MLP
evaluation callable.

### Pass Lifetime: The Reload Problem

The host creates a **fresh pass instance for every execution** via
`Activator.CreateInstance(type)`, binds resources and parameters, calls
`Execute`, and disposes it in a `finally`
(`src/Feather.Blender.RenderHost/ProjectPassAssembly.cs:211–234`). GPU-resident
history survives because the *pools* are owned one level up
(`ProjectPassAssembly.cs:89–95`), not the passes.

In `PROGRESSIVE` or `OFFLINE` mode that means a naive inference pass would
re-read the checkpoint file and re-upload every weight to the GPU on every
iteration. `RenderContext` exposes no project path and no cross-iteration cache
(`src/Feather/RenderGraph/RenderContext.cs:341–581`). This is the single most
important host-side gap for inference.

## (b) Required SDK Work

### P0 — Minimal Viable Platform Story

#### P0.1 A host-driven training job contract

**Do not put the loop in the SDK.** The existing platform design has the host own
the loop and project code provide types (`IRenderPass` +
`ProjectPassAssembly`). Training should mirror that exactly: the project declares
a job, the host drives it one step at a time, so the host owns cadence,
cancellation, event emission, and checkpoint timing. This avoids adding a
blocking `Train(epochs)` call that no host can interrupt.

New file: `src/Feather/NN/TrainingJob.cs`

```csharp
namespace Feather.NN;

/// <summary>Marks a project type the platform can drive as a training job.</summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class FeatherTrainerAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;
    public string? Name { get; set; }
    public string Category { get; set; } = "Training";
    public int Version { get; set; } = 1;
}

/// <summary>A project-authored training job driven one step at a time by a host.</summary>
public interface ITrainingJob : IDisposable
{
    /// <summary>Total planned steps, or 0 for open-ended training.</summary>
    int PlannedSteps { get; }

    /// <summary>Parameters the host may checkpoint. Must be stable after Initialize.</summary>
    IReadOnlyList<IParameter> Parameters { get; }

    void Initialize(TrainingContext context);

    /// <summary>Runs exactly one optimizer step and returns its report.</summary>
    TrainingStepReport Step(TrainingContext context);
}

/// <summary>One step's outcome. Loss is NaN when the job did not read it back.</summary>
public readonly record struct TrainingStepReport(
    int Step,
    float Loss,
    DispatchPath DispatchPath)
{
    public static TrainingStepReport Diverged(int step, float loss)
        => new(step, loss, DispatchPath.None);
}
```

`TrainingContext` is the counterpart to `RenderContext`: it is what makes the job
host-agnostic and gives it the project directory and cancellation.

```csharp
public sealed class TrainingContext
{
    /// <summary>Absolute project root, from the pass manifest's projectRoot.</summary>
    public string ProjectRoot { get; }

    /// <summary>Cooperative cancellation owned by the host.</summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>Zero-based index of the step about to run.</summary>
    public int Step { get; }

    /// <summary>Host-supplied hyperparameters, from graph node parameters.</summary>
    public bool TryGetSetting<T>(string name, out T value) where T : unmanaged;

    /// <summary>Resolves a project-relative path and creates parent directories.</summary>
    public string ResolveProjectPath(string relativePath);
}
```

`ResolveProjectPath` is deliberately on the context rather than a static helper so
the host controls the sandbox root. It must reject rooted paths and any path that
escapes `ProjectRoot` after normalization.

**Why `PlannedSteps` and not an epoch loop:** the AD path has no dataset concept
(`nn-status.md:71`), so "epoch" has no SDK meaning yet. Steps are the honest unit.
Epochs arrive with P1 datasets.

#### P0.2 Loss reporting without a per-step stall

`TrainingStep.Run()` blocks on `lossBuffer.ToArray()` every step
(`src/Feather/NN/Modules.cs:732`). At the host's reporting cadence (every N steps)
that stall is wasted on the N−1 steps whose loss nobody reads.

Three additive calls in `src/Feather/NN/Modules.cs` separate readback and queue completion:

```csharp
public sealed class TrainingStep<TKernel> : IDisposable
    where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
{
    /// <summary>
    /// Runs backward, gradient handoff, and the optimizer step without reading the
    /// loss buffer back. LastLoss is left unchanged. Prefer this on steps whose loss
    /// the caller will not report.
    /// </summary>
    public void RunWithoutLossReadback();

    /// <summary>
    /// Enqueues backward, gradient handoff, and the optimizer step without a loss
    /// readback or CPU completion wait. The host supplies a GPU.Queue fence.
    /// </summary>
    public void EnqueueWithoutLossReadback();

    /// <summary>
    /// Reads the loss buffer and returns the reduced scalar without running a step.
    /// Use after RunWithoutLossReadback on reporting steps.
    /// </summary>
    public float ReadLoss();
}
```

`Run()` keeps its exact current behavior and stays the default. The enqueue path is
validated against backend synchronization counters and queue-fence completion.

Also make device-side handoff public so a project can drive a custom AD kernel
without `TrainingStep`:

```csharp
// src/Feather/NN/Modules.cs — currently internal at line 609
public void Step<TKernel>(GpuADKernel<TKernel> adKernel)
    where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>;
```

This unblocks any project that needs several AD kernels per step (for example
separate losses per layer group) without falling back to the debug readback path.

A ranged readback on `GpuBuffer<T>` is the deeper fix and belongs in P1 — see
P1.4 — because `ToArray()` on a batch-sized loss buffer is a few kilobytes, not
the real bottleneck.

#### P0.3 Checkpoint I/O the platform can actually use

`Checkpoint` today is `Save(path, parameters)` / `Load(path, parameters)` with a
non-atomic `File.Create` (`src/Feather/NN/Checkpoint.cs:22`) and no metadata. For
a long-lived host writing checkpoints while a renderer may be reading them, that
is unsafe, and there is no way to know what a `.fthc` file contains without
already owning matching parameters.

Extend `src/Feather/NN/Checkpoint.cs`, keeping version 1 readable:

```csharp
public static class Checkpoint
{
    // Existing, unchanged:
    public static void Save(string path, IEnumerable<IParameter> parameters);
    public static void Load(string path, IEnumerable<IParameter> parameters);

    /// <summary>
    /// Writes to a temporary sibling file and atomically replaces the destination, so a
    /// concurrent reader never observes a partial checkpoint.
    /// </summary>
    public static void SaveAtomic(string path, IEnumerable<IParameter> parameters, CheckpointMetadata? metadata = null);

    /// <summary>Reads the header and parameter table without touching the GPU.</summary>
    public static CheckpointInfo Inspect(string path);

    /// <summary>
    /// Loads and reports what happened instead of silently skipping unmatched names.
    /// Today Load skips a name it cannot match (Checkpoint.cs:82-85).
    /// </summary>
    public static CheckpointLoadResult LoadStrict(string path, IEnumerable<IParameter> parameters);
}

/// <summary>Optional provenance written into a version 2 checkpoint.</summary>
public sealed record CheckpointMetadata(
    int Step,
    float Loss,
    string? ModelKind = null,
    IReadOnlyDictionary<string, string>? Tags = null);

public sealed record CheckpointInfo(
    uint Version,
    CheckpointMetadata? Metadata,
    IReadOnlyList<CheckpointEntryInfo> Entries);

public sealed record CheckpointEntryInfo(string FullName, TensorShape Shape);

public sealed record CheckpointLoadResult(
    IReadOnlyList<string> Loaded,
    IReadOnlyList<string> MissingFromFile,
    IReadOnlyList<string> UnusedInFile);
```

Format: bump `Version` to `2` and write `CheckpointMetadata` after the count field;
keep the version-1 read path so existing `.fthc` files load. `Inspect` is what lets
the Blender UI show "step 4200, loss 0.0031, 6 tensors" without loading anything.

`SaveAtomic` should use the same discipline the frame writer already uses — write
temp, close, `File.Move(temp, path, overwrite: true)` — mirroring the atomic
replace described in `docs/blender-render-host.md:208–211`.

#### P0.4 Weights to GPU for inference, cached across iterations

The primitive already exists (`Tensor<T>.AsReadOnlyBuffer()`). What is missing is
a way for a pass to (1) find the checkpoint and (2) avoid re-reading it every
frame, given passes are recreated per execution
(`ProjectPassAssembly.cs:211–234`).

New file: `src/Feather/NN/InferenceWeights.cs`

```csharp
namespace Feather.NN;

/// <summary>
/// A read-only, GPU-resident set of named weight tensors loaded from a checkpoint,
/// intended for binding into inference kernels and shaders.
/// </summary>
public sealed class InferenceWeights : IDisposable
{
    /// <summary>Loads every float tensor in the checkpoint by name.</summary>
    public static InferenceWeights Load(string path);

    public string SourcePath { get; }

    /// <summary>The checkpoint's file identity, for cache invalidation.</summary>
    public CheckpointStamp Stamp { get; }

    public CheckpointMetadata? Metadata { get; }

    public IReadOnlyCollection<string> Names { get; }

    public Tensor<float> this[string fullName] { get; }

    public bool TryGet(string fullName, out Tensor<float> tensor);

    /// <summary>Shader-facing view for binding into a generated kernel or shader.</summary>
    public ReadOnlyBuffer<float> Buffer(string fullName);

    public void Dispose();
}

/// <summary>Length plus last-write time, matching the host's file-signature idiom.</summary>
public readonly record struct CheckpointStamp(long Length, long LastWriteTicks)
{
    public static CheckpointStamp? TryRead(string path);
}
```

`CheckpointStamp` intentionally mirrors the host's existing
`FileSignature` shape (`src/Feather.Blender.RenderHost/RenderHostProgram.cs:168`)
so the reload check reuses a proven pattern.

Caching needs one host-side addition, because the SDK cannot own it — the pass is
gone before the next iteration. Add to `RenderContext`
(`src/Feather/RenderGraph/RenderContext.cs`):

```csharp
public sealed class RenderContext
{
    /// <summary>Absolute project root resolved from the manifest's projectRoot.</summary>
    public string ProjectRoot { get; }

    /// <summary>Resolves a project-relative path; rejects escapes from ProjectRoot.</summary>
    public string ResolveProjectPath(string relativePath);

    /// <summary>
    /// Returns weights for a project-relative checkpoint, loaded once and reused across
    /// iterations and pass instances. Reloads only when the file stamp changes. The
    /// host owns the lifetime; the pass must not dispose the result.
    /// </summary>
    public InferenceWeights GetOrLoadWeights(string projectRelativePath);
}
```

`GetOrLoadWeights` is backed by a new host-owned pool along`GraphTexturePool` and
`RasterTargetPool` in `ProjectPassAssembly` (`ProjectPassAssembly.cs:89–95`),
keyed by resolved path, holding a `CheckpointStamp` for invalidation. That
placement is what makes it survive across iterations. `ProjectRoot` is already
parsed by the host (`ProjectPassAssembly.cs:383–386`) and just needs to reach
`RenderContext`.

Convention: checkpoints live in **`Assets/Models/<name>.fthc`** under the project
root, committed with the project. `.feather/` is documented as ignored,
machine-local cache (`docs/blender-render-host.md:24–29`) and is the wrong place
for durable trained weights. Training writes to
`Assets/Models/<name>.fthc` via `SaveAtomic`, and the inference pass names the
same relative path as a `[Parameter]`.

#### P0.5 A reusable MLP: shader library plus AD loss kernel

There is no generic MLP anywhere — not in `src/Feather/NN` (only the forward-only
`NnLinearForwardKernel`, `DeviceOps.cs:400`), and not as a reusable type in tests,
where `NNReluMlpMeanLossKernel` and `NNSequentialReluMlpMeanLossKernel` are
declared per test file. Without this, "train an MLP and render it" requires every
user to hand-write both an AD kernel and a matching inference kernel and keep
their weight layouts in sync. That is the largest practical barrier.

New file: `src/Feather/NN/MlpShader.cs`

```csharp
namespace Feather.NN;

/// <summary>
/// Shader-side MLP evaluation over flat weight buffers. Layout is row-major per layer:
/// weight[outIndex * inputSize + inIndex], bias[outIndex]. Callables are source-available
/// so consuming projects can lower them.
/// </summary>
[ShaderLibrary]
public static class MlpShader
{
    /// <summary>One dense layer with ReLU, writing into a caller-owned scratch range.</summary>
    [Callable]
    public static void DenseRelu(
        ReadOnlyBuffer<float> weight, int weightOffset,
        ReadOnlyBuffer<float> bias, int biasOffset,
        ReadWriteBuffer<float> scratch, int inputOffset, int outputOffset,
        int inputSize, int outputSize);

    /// <summary>One dense layer with no activation.</summary>
    [Callable]
    public static void Dense(
        ReadOnlyBuffer<float> weight, int weightOffset,
        ReadOnlyBuffer<float> bias, int biasOffset,
        ReadWriteBuffer<float> scratch, int inputOffset, int outputOffset,
        int inputSize, int outputSize);

    /// <summary>Fixed 3-input, 1-output MLP with two hidden layers of hiddenSize.</summary>
    [Callable]
    public static float Evaluate3To1(
        float3 input,
        ReadOnlyBuffer<float> weights,
        ReadWriteBuffer<float> scratch, int scratchOffset,
        int hiddenSize);

    /// <summary>Flat element count for a packed 3→h→h→1 network, for buffer sizing.</summary>
    public static int PackedElementCount3To1(int hiddenSize);

    /// <summary>Scratch elements one lane needs for Evaluate3To1.</summary>
    public static int ScratchElementsPerLane3To1(int hiddenSize);
}
```

Two design decisions worth defending:

- **Packed single weight buffer, not one buffer per layer.** A single
  `ReadOnlyBuffer<float>` with documented offsets keeps binding counts low, keeps
  one `AD.Parameter` marker for the whole network, and matches what
  `SelfAttention` already does with its packed `[4, embeddingSize, embeddingSize]`
  weights (`SequenceModels.cs:85`). `PackedElementCount3To1` makes the layout
  computable from both host and shader side.
- **Fixed arity (`3To1`) rather than a fully generic loop.** The AD subset rejects
  `while`, and EasyGPU will not differentiate a parameter read at a
  variable-computed index (`EasyGPU/docs/autodiff.md:786`). A fixed-arity callable
  with counted `for` loops over `Uniform<int>` bounds is the shape known to
  lower — it is what the GPT kernel already does (`SequenceModels.cs:1276–1289`).
  Generic arity is P1 and must be proven against the adjoint generator, not
  assumed.

Alongside it, one concrete AD training kernel so nobody writes their own for the
common case:

```csharp
// src/Feather/NN/MlpTraining.cs — new file
[Kernel]
[AutoDiff]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct MlpRegression3To1LossKernel(
    ReadOnlyBuffer<float> inputs,     // 3 floats per sample
    ReadOnlyBuffer<float> targets,    // 1 float per sample
    ReadWriteBuffer<float> weights,   // packed, PackedElementCount3To1(hiddenSize)
    ReadWriteBuffer<float> scratch,   // ScratchElementsPerLane3To1 * count
    ReadWriteBuffer<float> loss,      // one per sample
    Uniform<int> hiddenSize,
    Uniform<float> lossScale) : IKernel1D
{
    public void Execute();
}
```

`ThreadGroupSize(1, 1, 1)` matches the existing AD-kernel idiom in
`docs/autodiff.md:34` and `samples/AdLinearRegression`. The `scratch` buffer is
the same technique `GptLanguageModelTrainer` uses (`SequenceModels.cs:514,529`).
One `AD.Parameter(weights[0])` covers the whole packed network, per
`native/feather_c_api.cpp:5516`.

Then a driver that wires kernel, optimizer, and checkpointing into an
`ITrainingJob`, so the common case is a few lines in a project:

```csharp
// src/Feather/NN/MlpTraining.cs
public sealed class MlpRegressionJob : ITrainingJob
{
    public MlpRegressionJob(
        int hiddenSize,
        ReadOnlySpan<float> inputs,
        ReadOnlySpan<float> targets,
        float learningRate = 0.01f,
        int seed = 1234);

    public int PlannedSteps { get; init; }
    public IReadOnlyList<IParameter> Parameters { get; }

    public void Initialize(TrainingContext context);
    public TrainingStepReport Step(TrainingContext context);
    public void Dispose();
}
```

#### P0.6 A training host the add-on can drive

New project: `src/Feather.Blender.TrainHost/` — deliberately a separate tool
rather than a mode of `Feather.Blender.RenderHost`, because a training run and a
viewport render have different lifetimes and must not contend for one process's
render loop.

It reuses the RenderHost's proven mechanics: `ProjectPassAssembly`-style
collectible `AssemblyLoadContext` loading against a manifest `buildId`, the
`WriteEvent(name, value)` stdout JSON writer
(`src/Feather.Blender.RenderHost/RenderHostProgram.cs:157`), `ProtocolJson.Options`
(`ProtocolJson.cs:5`), and `Console.CancelKeyPress` → `CancellationTokenSource`
(`RenderHostProgram.cs:27–33`).

```bash
dotnet tool run feather-blender-trainhost -- \
  --request .feather/cache/train.request.json
```

Its loop is the only place step counting, reporting cadence, checkpoint cadence,
and cancellation live:

```csharp
// src/Feather.Blender.TrainHost/TrainingRunner.cs
job.Initialize(context);
for (var step = 0; !token.IsCancellationRequested; step++)
{
    var report = job.Step(context);
    if (step % reportEverySteps == 0 || IsLastStep(step))
    {
        WriteEvent("loss", new TrainingProgressEvent(...));
    }

    if (checkpointEverySteps > 0 && step % checkpointEverySteps == 0)
    {
        Checkpoint.SaveAtomic(checkpointPath, job.Parameters,
            new CheckpointMetadata(step, report.Loss));
        WriteEvent("checkpoint", new CheckpointEvent(...));
    }

    if (!float.IsFinite(report.Loss)) { /* diverged: report and stop */ }
}
```

Divergence detection belongs here, not in every project, because
`samples/AdGptPoetDemo` proves users otherwise hand-roll it.

#### P0.7 Manifest and graph support for training nodes

The generator's manifest writer (`src/Feather.Generators/PassManifestWriter.cs:221`)
emits `passes` only. The host validates node `kind` against a closed string set —
`scene`, `pass`, `output`, `history-read`, `history-write`, `texture`, `camera`,
`object` (`src/Feather.Blender.RenderHost/RenderGraphDocument.cs:165`) — so a
training node cannot be represented today.

Two additive changes:

- `PassManifestWriter` emits a sibling `trainers` array for `[FeatherTrainer]`
  types, with the same `guid` / `typeName` / `parameters` shape as `passes`, so
  the add-on can populate a training panel from the same manifest and the same
  `buildId` validation applies.
- The train request schema is separate from the render request and is the
  TrainHost's own contract (see (c)), so `RenderGraphDocument`'s `kind` set is
  left alone for P0. Adding a `"trainer"` graph node kind is P1, and only if
  training should be a node in the render graph rather than a project-level job.
  Recommendation: keep it out of the render graph. A training run is not a frame.

### P1 — Breadth

#### P1.1 Datasets and batching

`nn-status.md:71` lists a data-loader framework as pending, and "epoch" has no SDK
meaning today. Minimum useful shape, in a new `src/Feather/NN/Dataset.cs`:

```csharp
public interface IDataset
{
    int Count { get; }
    int InputStride { get; }
    int TargetStride { get; }
    void WriteBatch(ReadOnlySpan<int> indices, Span<float> inputs, Span<float> targets);
}

public sealed class BatchSampler
{
    public BatchSampler(int count, int batchSize, int seed, bool shuffle = true);
    public int BatchesPerEpoch { get; }
    public void NextBatch(Span<int> indices);
}

/// <summary>Uploads one batch into the fixed device buffers an AD kernel already binds.</summary>
public sealed class DeviceBatchFeeder : IDisposable
{
    public DeviceBatchFeeder(GpuBuffer<float> inputs, GpuBuffer<float> targets, IDataset dataset);
    public void Upload(ReadOnlySpan<int> indices);
}
```

`DeviceBatchFeeder` matters because a `[Kernel]` struct's `Uniform<T>` values are
immutable after construction (`src/Feather/Resources/Uniform.cs:14`), which is why
`GptLanguageModelTrainer` constructs its AD kernel once and uploads into a
fixed-size token buffer per step (`SequenceModels.cs:531–546`, `UploadTokenBatch`).
Batching must follow that idiom: fixed shapes, upload per step, never rebuild the
kernel.

Once this lands, `TrainingContext` gains `Epoch` and `StepInEpoch`, and the loss
event gains `epoch`.

#### P1.2 Multi-loss and gradient accumulation

Today: exactly one `AD.Loss(float)`; non-scalar overloads exist to be rejected
(`src/Feather/AD/AD.cs:28–32`). Weighted multi-term losses must be summed into one
scalar inside the kernel — which is fine and should be **documented as the
supported pattern** rather than treated as a limitation.

Real P1 work is accumulating gradients across several dispatches before one
optimizer step, so batch size can exceed what one dispatch can hold:

```csharp
// src/Feather/NN/Modules.cs
public sealed class TrainingStep<TKernel>
{
    /// <summary>Adds this dispatch's gradients into parameter gradients without stepping.</summary>
    public void AccumulateGradients();

    /// <summary>Scales accumulated gradients, steps the optimizer, and zeroes them.</summary>
    public float StepAccumulated(float scale);
}
```

This needs a device-side add into `parameter.Gradient` — `CopyGradientToBuffer`
currently *replaces* (`src/Feather/AD/AD.cs:172`, native
`fe_kernel_reduce_ad_gradient_to_buffer`). Requires a native
accumulate-into-buffer variant. Scope this against the native bridge before
committing.

#### P1.3 AD breadth

Ranked by value to this platform:

1. **Multi-dimensional AD kernels.** `IKernel1D` only today
   (`src/Feather/AD/AD.cs:44`). Training an image-space loss (inverse rendering, a
   neural-radiance-style objective) wants `IKernel2D`. This is the single highest-value
   AD extension for a *renderer* platform.
2. **Chained AD across dispatches.** The real fix for deep models. Requires the
   adjoint generator to treat an intermediate buffer as both an output with an
   incoming adjoint and an input producing an outgoing one. Large native change in
   `EasyGPU/source/AD/`; scope separately.
3. **`while`/`break`/`continue`.** Needed for a ray-marched SDF loss with early
   exit. `EasyGPU/docs/autodiff.md:786` explains why `while` cannot be reversed
   without a trip count; a bounded `for` with a `float` mask is the workaround and
   should be documented as such.
4. **Texture gradients.** `nn.md:206` excludes them. Needed for learned textures
   and feature grids.

#### P1.4 Readback and interop ergonomics

- Ranged `GpuBuffer<T>.Read(int startIndex, Span<T> destination)`. `Upload`
  already takes a start index (`src/Feather/Resources/GpuBuffer.cs:104`); `Read`
  does not (line 135).
- Async or fenced readback, so loss reporting never stalls the training
  dispatch. Note `docs/blender-render-host.md:290` already flags synchronous
  readback as a deliberate first-integration choice.
- Device-to-device buffer copy, so a training job can publish weights to an
  inference buffer without a host round trip. This is what would eventually let
  training and rendering share one process.
- `Tensor<float>` → `GpuTexture2D<float, float>` upload for `R32Float` weight or
  feature-grid lookup with hardware filtering.
- Make `TensorView<T>`'s `Offset` meaningful by having `NnDeviceOps` accept
  views, or delete the type. Carrying an offset no op honors is a trap.

#### P1.5 Optimizer and model breadth

Non-float parameters (`Optimizer` rejects them at
`src/Feather/NN/Modules.cs:515`), LR schedules, gradient-norm clipping across all
optimizers (only `Adam` takes `gradientClip` today, line 847), and optimizer
state in checkpoints so a run resumes exactly. The last one matters most for a
long training session in Blender: today `Checkpoint` saves weights only, so
resuming loses Adam's moments and the step count.

## (c) Blender-Side Integration Contract

### Transport

Reuse the RenderHost's proven mechanism exactly: a JSON request file that Blender
writes to a temp path and atomically renames, and newline-delimited JSON events on
the child process's stdout, with errors on stderr
(`docs/blender-render-host.md:56–60`,
`src/Feather.Blender.RenderHost/RenderHostProgram.cs:157–166`). No sockets, no new
IPC layer.

### Train Request V1

`.feather/cache/train.request.json`, paths relative to the request file:

```json
{
  "schemaVersion": 1,
  "requestId": 7,
  "generationId": "5ebc93da-b905-4f44-8eda-68968bb6ba2f",
  "manifestPath": "../../Generated/pass-manifest.json",
  "trainerGuid": "b2c1f0a4-3d5e-4a7b-9c8d-1e2f3a4b5c6d",
  "typeName": "MyProject.Training.SdfMlpJob",
  "parameters": [
    { "name": "HiddenSize", "value": 32 },
    { "name": "LearningRate", "value": 0.01 }
  ],
  "plannedSteps": 5000,
  "reportEverySteps": 25,
  "checkpointEverySteps": 500,
  "checkpointPath": "Assets/Models/sdf-mlp.fthc",
  "resumeFromCheckpoint": true
}
```

`generationId` and `manifestPath` carry the same meaning and the same `buildId`
validation as the render request, so a stale build cannot be trained against
(`docs/blender-render-host.md:91–97`). `checkpointPath` is project-relative and
resolved against the manifest's `projectRoot`.

### Events The Add-On Consumes

All events keep the existing envelope, `{ "event": <name>, "value": { ... } }`.

| Event | When | Payload |
| --- | --- | --- |
| `ready` | Once at startup | `{ requestPath, trainerTypeName, plannedSteps }` |
| `loss` | Every `reportEverySteps`, plus first and last step | `{ step, plannedSteps, loss, dispatchPath, stepsPerSecond, elapsedMilliseconds }` |
| `checkpoint` | After each successful `SaveAtomic` | `{ step, loss, path, absolutePath, sizeInBytes }` |
| `finished` | Training completed, cancelled, or diverged | `{ step, finalLoss, reason, checkpointPath }` where `reason` is `completed` \| `cancelled` \| `diverged` \| `failed` |
| `error` | Exception | `{ error, message }` — identical to the render host's shape (`RenderHostProgram.cs:163`) |

Contract points the add-on can rely on:

- `loss` events are monotonic in `step` and never dropped for the first or last
  step, so a graph always has endpoints.
- `finished` is emitted exactly once and is always the last event before exit.
- `dispatchPath` on a `loss` event lets the UI surface a fallback route the same
  way the render panel does; anything other than `TypedEasyGpu` is a red flag.
- A `checkpoint` event's `absolutePath` is safe to read immediately — the write
  was atomic.
- Cancellation is `SIGINT` / `Console.CancelKeyPress`, matching the render host.
  The add-on's Cancel button sends it and waits for `finished`.

### Add-On Responsibilities

- Populate a training panel from the manifest's new `trainers` array; parameters
  come from the same `[Parameter]` metadata shape as passes.
- Own the loss graph. The host streams points; it does not aggregate history.
- Show `Checkpoint.Inspect` output (step, loss, tensor count) for the checkpoint an
  inference pass references, so a user can tell trained from untrained.
- Do not restart the RenderHost when a checkpoint is written. The inference pass
  picks it up on its next iteration via `GetOrLoadWeights` stamp invalidation. A
  viewport tag-redraw is enough.

## (d) End-To-End Example: Train An MLP As An SDF, Then Render It

The example to ship as `samples/NeuralSdf`. It is chosen because it exercises the
whole chain — AD training, checkpointing, weight upload, in-kernel inference —
while staying inside the AD subset, and because `samples/SdfRenderer` already
establishes the SDF-in-a-kernel idiom (`samples/SdfRenderer/Program.cs:262`).

### Project Layout

```
MyProject/
  MyProject.csproj
  Training/
    SdfMlpJob.cs            # [FeatherTrainer] ITrainingJob
    SdfMlpLossKernel.cs     # [AutoDiff] IKernel1D
  Passes/
    NeuralSdfPass.cs        # [FeatherPass] IComputePass
    NeuralSdfKernel.cs      # [Kernel] IKernel2D, inference only
  Assets/Models/
    sdf-mlp.fthc            # written by SaveAtomic, committed
  Generated/
    pass-manifest.json      # passes + trainers
```

### Step 1 — Ground Truth On The Host

Sample N points in `[-1, 1]^3` and evaluate an analytic SDF (sphere ∪ box, matching
`samples/SdfRenderer`) to produce `inputs[3N]` and `targets[N]`. Plain .NET;
`nn-status.md:27` already establishes dataset handling as host code.

### Step 2 — The AD Loss Kernel

Structurally identical to `MlpRegression3To1LossKernel` from P0.5: read three
inputs for lane `i`, run 3→h→h→1 through the packed weight buffer using counted
`for` loops over `Uniform<int>` bounds and a per-lane `scratch` range, compute
`(prediction − target)^2 * lossScale`, write it to `loss[i]`, then
`AD.Parameter(weights[0])` and `AD.Loss(l)`.

Constraints this respects, by construction:

- 1D kernel, one scalar loss.
- No `while`, no `break`.
- Parameter reads are at counted-loop indices, not host-variable indices — the
  case `EasyGPU/docs/autodiff.md:786` warns about.
- One marker for the whole packed buffer (`native/feather_c_api.cpp:5516`).

Verification gate before anything else is built on it: `GetBackwardGLSL()` returns
non-empty, `LastDispatchPath == TypedEasyGpu`, and gradients match central finite
differences — the discipline
`tests/Feather.AD.Tests/ADNumericalCorrectnessTests.cs` already applies.

### Step 3 — The Training Job

```csharp
[FeatherTrainer("b2c1f0a4-3d5e-4a7b-9c8d-1e2f3a4b5c6d", Name = "SDF MLP", Category = "Training")]
public sealed class SdfMlpJob : ITrainingJob
{
    [Parameter("...")] public int HiddenSize { get; set; } = 32;
    [Parameter("...")] public float LearningRate { get; set; } = 0.01f;

    private TrainingStep<SdfMlpLossKernel>? step;
    private Parameter<float>? weights;

    public int PlannedSteps { get; set; } = 5000;
    public IReadOnlyList<IParameter> Parameters => weights is null ? [] : [weights];

    public void Initialize(TrainingContext context)
    {
        // Sample ground truth, allocate device buffers, Xavier-init packed weights,
        // build the AD kernel once, wrap it in TrainingStep with an Adam optimizer.
    }

    public TrainingStepReport Step(TrainingContext context)
    {
        // Cheap steps skip the readback; the host asks for loss on reporting steps.
        step!.RunWithoutLossReadback();
        return new TrainingStepReport(context.Step, step.ReadLoss(), step.LastDispatchPath);
    }

    public void Dispose() { /* step, buffers, weights */ }
}
```

The host drives it: `--request` names this trainer, the TrainHost calls `Step` in a
loop, emits a `loss` event every 25 steps, and calls
`Checkpoint.SaveAtomic("Assets/Models/sdf-mlp.fthc", job.Parameters, metadata)`
every 500. Blender draws the curve live and the Cancel button raises the token.

### Step 4 — The Inference Pass

```csharp
[FeatherPass("...", Name = "Neural SDF", Category = "Compute")]
public sealed class NeuralSdfPass : IComputePass
{
    [Input("...")] public CameraHandle Camera { get; init; }
    [Output("...", Format = TextureFormat.Rgba8)] public TextureHandle Color { get; init; }

    [Parameter("...")] public string ModelPath { get; set; } = "Assets/Models/sdf-mlp.fthc";
    [Parameter("...")] public int HiddenSize { get; set; } = 32;

    public void Execute(RenderContext context)
    {
        // Host-cached: loaded once, reused across iterations and pass instances,
        // reloaded only when the checkpoint's stamp changes. Not disposed here.
        var weights = context.GetOrLoadWeights(ModelPath);
        var camera = context.GetCamera(Camera);
        var color = context.GetOrCreateTexture<Rgba8, Rgba8>(Color, PixelFormat.Rgba8);

        using var scratch = GPU.CreateBuffer<float>(
            context.Width * context.Height * MlpShader.ScratchElementsPerLane3To1(HiddenSize));

        var path = GPU.DispatchAndGetPath(
            new NeuralSdfKernel(
                weights.Buffer("mlp.weights"),
                scratch.AsReadWrite(),
                color.AsReadWrite(),
                new Uniform<int>(HiddenSize),
                new Uniform<float4x4>(camera.ViewProjection)),
            new int2(context.Width, context.Height));

        context.SetColorOutput(Color, color, path);
    }
}
```

`NeuralSdfKernel` is an `IKernel2D` that ray-marches, calling
`MlpShader.Evaluate3To1(p, weights, scratch, laneOffset, hiddenSize)` in place of
`SdfRenderer`'s analytic `Sdf` (`samples/SdfRenderer/Program.cs:262`), shades from
a finite-difference normal, and writes RGBA8. Being an ordinary `[Kernel]` — no
`[AutoDiff]` — it is free of the AD subset restrictions and may use `while`,
`break`, and whatever else the compute subset allows.

### What This Proves

- Training runs in a long-lived host process with live loss and working cancellation.
- Checkpoints land in the project directory, atomically, with step and loss metadata.
- Trained weights reach a rendering kernel as an ordinary `ReadOnlyBuffer<float>`,
  loaded once rather than per frame.
- Train and render coexist: retraining rewrites the checkpoint, and the viewport
  picks it up on its next iteration with no host restart.
- Both halves of the user's requirement — rendering and the supporting NN
  infrastructure — live in the same project and the same platform.

### Per-Lane Scratch: The Sharp Edge

`Evaluate3To1` needs `hiddenSize` floats of working space per lane, and at
1920×1080 with `hiddenSize: 32` that is roughly 265 MB. Options, and the
recommendation:

- Per-lane global scratch: simple, correct, memory-hungry. Fine for a first sample
  at preview resolution.
- `SharedMemory<float>` per workgroup: far smaller, but sized at compile time
  (`docs/csharp-subset.md:212`), so `HiddenSize` becomes a compile-time constant
  rather than a `[Parameter]`.
- Fixed-size registers via `GpuArrayN<T>` in a `[GpuStruct]`
  (`docs/csharp-subset.md:160`): fastest, no scratch buffer, but fixes the network
  width at compile time.

Recommendation: ship the sample with per-lane global scratch for clarity, and file
the register-based variant as a follow-up. Do not let this choice block P0 —
but do measure it, because it may dominate the pass's cost.

## (e) Risks And Open Questions

### Risks

**R1 — The MLP AD kernel may not lower.** The whole P0 story rests on
`MlpRegression3To1LossKernel` generating a valid adjoint. The GPT kernel proves
that nested counted loops over weight buffers *can* work
(`SequenceModels.cs:1276–1289`), but EasyGPU's own limits list variable buffer
indexing as non-differentiable (`EasyGPU/docs/autodiff.md:786`), and the boundary
between "counted-loop index" and "variable index" is not documented precisely.
*Mitigation:* build this kernel and validate it against finite differences
**first**, before any host or add-on work. If it fails, everything downstream
changes shape. This is the go/no-go gate.

**R2 — Two processes contending for the GPU.** A TrainHost and a RenderHost both
holding device contexts, with a Blender viewport also drawing, may starve the
viewport or hit driver-level allocation limits. *Mitigation:* measure early on the
target backends; if it is bad, the add-on serializes them (pause viewport rendering
while training) before any architectural change. Note Feather does not choose a
backend at runtime (`docs/support-status.md:51`), so both processes are the same
backend by construction.

**R3 — Per-frame checkpoint reload if `GetOrLoadWeights` is missed.** Because
passes are recreated per execution (`ProjectPassAssembly.cs:211–234`), the obvious
implementation of an inference pass calls `Checkpoint.Load` in `Execute` and
silently re-reads and re-uploads every frame. *Mitigation:* ship
`GetOrLoadWeights` in the same change as the sample, and make the docs never show
`Checkpoint.Load` inside a pass.

**R4 — Checkpoint format churn.** `nn-status.md:73` explicitly declines format
stability guarantees, and P0.3 bumps to version 2. *Mitigation:* keep the
version-1 read path (the loader already checks version at
`Checkpoint.cs:59`), and have `Inspect` fail with a clear message on a future
version rather than misreading it.

**R5 — Scratch memory at full resolution.** See the sharp edge above. Could make
the sample look bad on modest GPUs. *Mitigation:* default the sample to a reduced
`resolutionScale`, and report scratch size in the pass's diagnostics.

**R6 — Silent skips in `Checkpoint.Load`.** Today an unmatched name is skipped
without a word (`Checkpoint.cs:82–85`), so a renamed layer yields a
partially-loaded model that renders plausible garbage. *Mitigation:* `LoadStrict`,
and have `InferenceWeights.Load` use it.

### Open Questions

1. **Should training be a graph node or a project-level job?** This design says
   project-level job, out of the render graph — a training run is not a frame, and
   the graph's `kind` set (`RenderGraphDocument.cs:165`) plus its per-request
   execution model do not fit a 5000-step run. If the product wants training
   visible as a node, the request/event contract still holds but the add-on needs
   a different UI model. **Needs a product decision before P0.6.**
2. **One process or two?** Two (separate TrainHost) is proposed for lifetime
   isolation, at the cost of R2 and of not being able to hand weights from training
   to rendering without going through disk. A single process would allow a
   device-to-device handoff (P1.4) and live "watch it learn" rendering. Two first;
   revisit after R2 is measured.
3. **Where do checkpoints live?** `Assets/Models/*.fthc` is proposed —
   project-relative and committed — because `.feather/` is documented as ignored
   machine-local cache (`docs/blender-render-host.md:24–29`). Confirm this against
   how the Blender fork expects project assets to be committed.
4. **Should optimizer state be checkpointed?** Not in P0, which means a resumed run
   restarts Adam's moments and step count. Acceptable for a preview; needs a
   decision if long multi-session training is a real workflow.
5. **Is `IKernel2D` AD the actual priority?** For a *renderer* platform, an
   image-space loss (inverse rendering, differentiable shading) may matter more
   than deep-model support. It is listed first in P1.3 for that reason, but the
   ranking deserves a product opinion.
6. **How does the add-on know a pass depends on a checkpoint?** In this design the
   path is just a `[Parameter]` string, so the UI cannot tell that a pass consumes
   a model. A dedicated socket kind or a manifest annotation would let Blender
   wire trainer output to pass input explicitly. Worth resolving alongside
   question 1.

## Related Docs

- [Neural Networks](nn.md), [Feather.NN Status](nn-status.md)
- [Automatic Differentiation](autodiff.md), [AD Internals And Coverage](ad-implementation-note.md)
- [Blender RenderHost](blender-render-host.md)
- [C# Shader Subset](csharp-subset.md), [Support Status](support-status.md)
