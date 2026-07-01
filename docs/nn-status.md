# Feather.NN Status

This page explains what `Feather.NN` does on the GPU today, where it intentionally crosses the host boundary, and which parts are preview-quality. Read [Neural Networks](nn.md) first for the user guide.

`Feather.NN` is useful for small GPU-resident experiments, tests, and sample trainers. It is not a general replacement for PyTorch, TensorFlow, or a production ML runtime.

## GPU-Native Today

The following surfaces are backed by GPU buffers and native dispatch:

- `Tensor<T>` wraps a `GpuBuffer<T>`.
- `Parameter<float>` owns value and gradient tensors.
- `TensorOps` elementwise operations dispatch device work.
- `Linear`, `Embedding`, `LayerNorm`, `BatchNorm1D`, and activation modules use device operations.
- `SGD`, `RMSProp`, `Adam`, and `AdamW` keep optimizer state in GPU tensors.
- `TrainingStep<TKernel>` runs Feather AD, reduces gradients into parameter gradient buffers, and steps the optimizer.
- Sample trainers for AD/Transformer/GPT paths exercise GPU-native training kernels.

## Explicit Host Boundaries

Some boundaries are intentionally visible:

- Scalar-returning loss helpers read back one value.
- Checkpoints save/load parameter buffers through host memory.
- Progress logging reads scalar losses.
- Some sequence-model inference helpers are named `*Host` because they read buffers back and run host-side logic.
- Dataset/token handling remains ordinary .NET code.

These boundaries are not hidden because users need to understand synchronization and performance costs.

## AD Integration

NN training uses Feather AD rather than dynamic autograd over module calls. The recommended training shape is:

1. Own parameters with `Parameter<float>`.
2. Write a generated `[AutoDiff]` loss kernel over the relevant parameter buffers.
3. Add gradient aliases if kernel resource names differ from parameter names.
4. Create `TrainingStep<TKernel>`.
5. Run the step, which handles backward, gradient handoff, and optimizer update.

This makes the training boundary explicit and keeps the generated GPU code inspectable.

## Proven By Tests And Samples

Representative coverage includes:

- Parameter ownership and zero-grad.
- `Sequential` parameter qualification.
- `Linear` vector and batch forward paths.
- Tensor ops and losses.
- Optimizer state and step behavior.
- AD gradient handoff into optimizer buffers.
- Checkpoint save/load.
- Transformer/GPT demo surfaces.

Representative samples:

- `samples/AdLinearRegression`
- `samples/AdTransformer`
- `samples/AdGptDemo`
- `samples/AdGptPoetDemo`
- `samples/ProfilerSuite`

## Unsupported Or Pending

- Dynamic graph autograd over arbitrary `Module.Forward` calls.
- General GPU-native autoregressive inference.
- Non-float optimizers.
- Large-model training infrastructure.
- Distributed training.
- Dataset/data-loader framework.
- Mixed precision training policy.
- Production checkpoint format stability guarantees.

## Gates For NN Changes

Before treating an NN change as complete:

- Run managed NN tests.
- Run AD tests when gradient handoff or training steps change.
- Run relevant samples when public behavior changes.
- Check `ProfilerSuite` when performance-sensitive paths change.
- Keep host-boundary naming explicit.

## Related Docs

- [Neural Networks](nn.md)
- [Automatic Differentiation](autodiff.md)
- [API: Neural Networks](api/nn.md)
- [AD Internals And Coverage](ad-implementation-note.md)
