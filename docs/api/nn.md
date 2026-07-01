# API Reference: Neural Networks

## Purpose

`Feather.NN` provides GPU-buffer-backed tensors, modules, losses, optimizers, training steps, and checkpoints for small explicit training workloads.

## Tensors And Parameters

| API | Purpose |
| --- | --- |
| `Tensor<T>` | Shape plus backing `GpuBuffer<T>`. |
| `TensorShape` | Validated dimensions and element count. |
| `TensorView<T>` | View over tensor, shape, and offset. |
| `Tensor2D<T>` | Convenience 2D tensor wrapper. |
| `Parameter<T>` | Value tensor plus gradient tensor and stable names. |
| `IParameter` | Optimizer/checkpoint abstraction. |
| `ParameterInitializers` | Xavier and constant parameter factories. |
| `ParameterGroup` | Optimizer parameter subset with optional overrides. |

Typical parameter creation:

```csharp
using var weight = ParameterInitializers.XavierParameter(
    "weight",
    new TensorShape(2, 3),
    fanIn: 3,
    fanOut: 2,
    seed: 123);
```

## Modules

| Module | Purpose |
| --- | --- |
| `Module` | Base class with `Parameters` and `QualifyParameters`. |
| `Linear` | Affine transform over last dimension. |
| `Embedding` | Integer index to learned vector table. |
| `LayerNorm` | Per-vector normalization. |
| `BatchNorm1D` | 1D batch normalization. |
| `Sequential` | Composes modules and qualifies parameter names. |
| `ReLU`, `Sigmoid`, `Tanh`, `SiLU` | Activation modules. |
| `Softmax`, `LogSoftmax` | Probability/log-probability activations. |

```csharp
using var model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
using Tensor<float> y = model.Forward(x);
```

## Tensor Ops And Losses

`TensorOps`:

- `Add`, `Subtract`, `Multiply`, `Divide`
- Scalar overloads for arithmetic.
- `Copy`
- `Fill`
- `Softmax`
- `LogSoftmax`

`Losses`:

- `MeanSquaredErrorTensor`
- `MeanSquaredError`
- `MeanAbsoluteErrorTensor`
- `MeanAbsoluteError`
- `CrossEntropyTensor`
- `CrossEntropy`
- `CrossEntropyFromLogitsTensor`
- `CrossEntropyFromLogits`

`CrossEntropyLoss` wraps cross-entropy calls in an object API.

## Optimizers

| Optimizer | Notes |
| --- | --- |
| `SGD` | Learning rate, momentum, weight decay. |
| `RMSProp` | Learning rate, alpha, epsilon, weight decay. |
| `Adam` | Learning rate, betas, epsilon, weight decay, gradient clip, parameter groups. |
| `AdamW` | Adam with decoupled weight decay style. |

Optimizers expose `Step()`, `ZeroGrad()`, and `Step(GradientSet)` for AD handoff.

## Training Steps

`TrainingStep<TKernel>` connects an AD kernel, parameters, optimizer, loss buffer, and dispatch count:

| API | Purpose |
| --- | --- |
| `Create(kernel, parameters, optimizer, lossBuffer, count)` | Creates the step wrapper. |
| `Run()` | Runs backward, gradient handoff, optimizer step, and loss readback. |
| `LastDispatchPath` | Last AD dispatch route. |
| `GradientsMaterialized` | Whether gradients were read back for fallback/debug. |
| `LastLoss` | Last scalar loss readback. |

## Checkpoints

```csharp
Checkpoint.Save("model.fthc", model.Parameters);
Checkpoint.Load("model.fthc", model.Parameters);
```

Checkpoints currently target named float parameters.

## Sequence Models

Preview helpers include `PositionalEmbedding`, `SelfAttention`, `TransformerBlock`, `GptLanguageModel`, `SelfAttentionClassifier`, and sample trainers. Host-named inference helpers intentionally cross the host boundary.

## Host Vs Shader

`Feather.NN` is a host-side helper layer over GPU buffers and generated kernels. Module APIs are called from normal .NET code. Training uses generated AD kernels for differentiable work rather than a dynamic host-side autograd graph.

## Lifetime And Errors

- `Tensor<T>`, `Parameter<T>`, `Module`, `Optimizer`, and `TrainingStep<TKernel>` implementations are disposable where they own buffers/native state.
- Optimizers currently target `Parameter<float>`.
- Scalar-returning losses and checkpoints intentionally cross the host boundary.
- Shape mismatches throw managed exceptions before dispatch.

## Guide

See [Neural Networks](../nn.md) and [Feather.NN Status](../nn-status.md).

## Samples And Tests

- `samples/AdLinearRegression`
- `samples/AdTransformer`
- `samples/AdGptDemo`
- `samples/AdGptPoetDemo`
- `tests/Feather.NN.Tests/NNSurfaceTests.cs`
- `tests/Feather.Integration.Tests/NNTrainingIntegrationTests.cs`
