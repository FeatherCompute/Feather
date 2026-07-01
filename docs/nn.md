# Neural Networks

`Feather.NN` is a preview helper layer for small GPU-resident models and experiments. It provides tensors backed by `GpuBuffer<T>`, trainable parameters, modules, tensor operations, losses, optimizers, checkpoints, and AD-backed training helpers.

It is not a PyTorch clone. There is no dynamic autograd graph over arbitrary `Module.Forward` calls. Training today is explicit: write or use a generated `[AutoDiff]` loss kernel, mark parameters/loss, then let `TrainingStep<TKernel>` hand gradients to optimizers.

![Cornell box rendered with Feather](img/cornell-box.png)

## Tensors

```csharp
using Feather;
using Feather.NN;

using var input = new Tensor<float>(
    new TensorShape(4, 3),
    GPU.CreateBuffer<float>(new float[12]));
```

`Tensor<T>` is a shape plus a `GpuBuffer<T>`.

| API | Purpose |
| --- | --- |
| `Tensor<T>.Shape` | Logical dimensions. |
| `Tensor<T>.Buffer` | Backing GPU buffer. |
| `Tensor<T>.Rank` | Number of dimensions. |
| `Tensor<T>.ElementCount` | Total logical element count. |
| `AsReadOnlyBuffer()` | Shader-facing read-only buffer view. |
| `AsReadWriteBuffer()` | Shader-facing read-write buffer view. |

`TensorShape` validates positive dimensions and computes `ElementCount`.

## Parameters

```csharp
using var parameter = ParameterInitializers.XavierParameter(
    "weight",
    new TensorShape(2, 3),
    fanIn: 3,
    fanOut: 2,
    seed: 123);
```

`Parameter<T>` pairs a value tensor and a gradient tensor:

| API | Purpose |
| --- | --- |
| `Name` | Stable parameter name. |
| `FullName` | Qualified module path after composition. |
| `GradientNames` | Names accepted when copying AD gradients. |
| `Value` | Learnable tensor. |
| `Gradient` | Gradient tensor. |
| `ZeroGrad()` | Clears the gradient tensor on the device for float parameters. |
| `AddGradientAlias(name)` | Accepts an AD gradient name that differs from the parameter name. |

## Modules

Core modules:

- `Linear`
- `Embedding`
- `LayerNorm`
- `BatchNorm1D`
- `ReLU`, `Sigmoid`, `Tanh`, `SiLU`
- `Softmax`, `LogSoftmax`
- `Sequential`

Example:

```csharp
using var layer = new Linear(inputSize: 3, outputSize: 2);
using Tensor<float> output = layer.Forward(input);
```

`Sequential` qualifies repeated module parameter names deterministically:

```csharp
using var model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
foreach (var parameter in model.Parameters)
{
    Console.WriteLine(parameter.FullName);
}
```

Expect names such as `linear0.weight`, `linear0.bias`, `linear1.weight`, and `linear1.bias`.

## Tensor Operations And Losses

Device-backed operations:

```csharp
using Tensor<float> sum = TensorOps.Add(left, right);
using Tensor<float> scaled = TensorOps.Multiply(input, 0.5f);
TensorOps.Fill(parameter.Gradient, 0.0f);
TensorOps.Copy(source, destination);
```

Losses:

```csharp
using Tensor<float> mseTensor = Losses.MeanSquaredErrorTensor(prediction, target);
float mse = Losses.MeanSquaredError(prediction, target);

using Tensor<float> ce = Losses.CrossEntropyFromLogitsTensor(logits, labels);
float ceValue = Losses.CrossEntropyFromLogits(logits, labels);
```

Methods returning `Tensor<float>` keep the scalar result on the device. Scalar-returning overloads intentionally read back one value for convenience.

## Optimizers

Available optimizers:

- `SGD`, with optional momentum and weight decay.
- `RMSProp`.
- `Adam`.
- `AdamW`.

```csharp
using var optimizer = new Adam(model.Parameters, learningRate: 0.001f);
optimizer.ZeroGrad();
optimizer.Step();
```

Optimizer state such as momentum and Adam moments is stored in GPU tensors. Optimizers currently support `Parameter<float>`.

Parameter groups let you override learning rate or weight decay for subsets:

```csharp
using var optimizer = new Adam(
[
    new ParameterGroup("backbone", backbone.Parameters, learningRate: 0.0005f),
    new ParameterGroup("head", head.Parameters, learningRate: 0.001f, weightDecay: 0.01f)
]);
```

## AD-Backed Training

For trainable modules, write a generated `[AutoDiff]` loss kernel over parameter buffers and wrap it in `TrainingStep<TKernel>`.

```csharp
using var optimizer = new SGD([weight, bias], learningRate: 0.05f);
using var step = TrainingStep<LinearRegressionAdKernel>.Create(
    new LinearRegressionAdKernel(
        x.AsReadOnly(),
        y.AsReadOnly(),
        weight.Value.Buffer.AsReadWrite(),
        bias.Value.Buffer.AsReadWrite(),
        loss.AsReadWrite()),
    [weight, bias],
    optimizer,
    loss,
    count: 1);

float lossValue = step.Run();
```

`TrainingStep<TKernel>`:

- Runs the native AD backward pass.
- Matches native gradient names to `Parameter<float>.GradientNames`.
- Reduces AD gradient storage into parameter gradient buffers on the device.
- Runs the optimizer update.
- Exposes `LastDispatchPath`, `GradientsMaterialized`, and `LastLoss`.

Use `parameter.AddGradientAlias("kernelResourceName")` when the AD kernel's resource name differs from the parameter name.

## Checkpoints

```csharp
Checkpoint.Save("model.fthc", model.Parameters);
Checkpoint.Load("model.fthc", model.Parameters);
```

Checkpoints currently save and load named float parameters.

## Sequence Models

Feather includes preview sequence-model helpers used by samples:

- `PositionalEmbedding`
- `SelfAttention`
- `TransformerBlock`
- `GptLanguageModel`
- `SelfAttentionClassifier`
- Built-in trainers for the sample attention classifier and GPT language model.

Current public inference helpers for Transformer/GPT-style models are intentionally named `*Host` because they read buffers back and run host-side inference math. GPU-native training kernels exist for the sample trainers, but general GPU-native autoregressive inference is future work.

## Samples

| Sample | Shows |
| --- | --- |
| `AdLinearRegression` | Smallest AD-backed optimizer loop. |
| `AdTransformer` | Transformer-style training path. |
| `AdGptDemo` | GPT language-model demo surface. |
| `AdGptPoetDemo` | Larger text-generation sample. |
| `ProfilerSuite` | NN timing and gradient materialization. |

## Current Limits

- No dynamic graph autograd over arbitrary `Module.Forward` calls.
- NN optimizers support `Parameter<float>`.
- Trainable module paths require generated AD loss kernels or built-in trainers.
- Vector losses are not supported.
- Texture, struct, graphics, and matrix gradients are not NN training surfaces.
- Scalar progress logging and checkpointing intentionally cross the host boundary.

For deeper status details, read [Feather.NN Status](nn-status.md).

## API Reference

See [API: Neural Networks](api/nn.md).
