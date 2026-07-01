using Feather;
using Feather.NN;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

const int steps = 8;
const float learningRate = 0.05f;

using var xBuffer = GPU.CreateBuffer<float>([2.0f], BufferAccess.ReadOnly);
using var yBuffer = GPU.CreateBuffer<float>([5.0f], BufferAccess.ReadOnly);
using var weight = new Parameter<float>(
    "weight",
    new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>([-0.5f], BufferAccess.ReadWrite), requiresGrad: true),
    new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>(1, BufferAccess.ReadWrite)));
weight.AddGradientAlias("w");
using var bias = new Parameter<float>(
    "bias",
    new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>([0.0f], BufferAccess.ReadWrite), requiresGrad: true),
    new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>(1, BufferAccess.ReadWrite)));
bias.AddGradientAlias("b");
using var lossBuffer = GPU.CreateBuffer<float>(1, BufferAccess.ReadWrite);
var optimizer = new SGD([weight, bias], learningRate: learningRate);
using var trainingStep = TrainingStep<LinearRegressionAdKernel>.Create(
    new LinearRegressionAdKernel(
        xBuffer.AsReadOnly(),
        yBuffer.AsReadOnly(),
        weight.Value.Buffer.AsReadWrite(),
        bias.Value.Buffer.AsReadWrite(),
        lossBuffer.AsReadWrite()),
    [weight, bias],
    optimizer,
    lossBuffer,
    count: 1);

Console.WriteLine("AD Linear Regression");
Console.WriteLine("step loss       w        b");

for (var step = 0; step < steps; step++)
{
    var loss = trainingStep.Run();

    var w = weight.Value.Buffer.ToArray()[0];
    var b = bias.Value.Buffer.ToArray()[0];

    Console.WriteLine($"{step,4} {loss,8:F5} {w,8:F4} {b,8:F4}");
}

var finalW = weight.Value.Buffer.ToArray()[0];
var finalB = bias.Value.Buffer.ToArray()[0];
Console.WriteLine($"final w={finalW:F4}, b={finalB:F4}");
Console.WriteLine($"dispatch={trainingStep.LastDispatchPath}, gradients materialized={trainingStep.GradientsMaterialized}");

[Kernel]
[AutoDiff]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct LinearRegressionAdKernel(
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> w,
    ReadWriteBuffer<float> b,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        var i = ThreadIds.X;
        var weight = w[0];
        var bias = b[0];
        var pred = weight * x[i] + bias;
        var error = pred - y[i];
        var l = error * error;

        loss[i] = l;
        ADMarker.Parameter(w[0]);
        ADMarker.Parameter(b[0]);
        ADMarker.Loss(l);
    }
}
