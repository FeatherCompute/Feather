using Feather.Math;
using Feather.Resources;

namespace Feather.NN;

internal enum NnElementwiseBinaryOp
{
    Add,
    Subtract,
    Multiply,
    Divide
}

public enum NnActivationKind
{
    ReLU,
    Sigmoid,
    Tanh,
    SiLU,
    Softmax,
    LogSoftmax
}

public static class NnDispatchTrace
{
    private static readonly List<string> operations = [];

    public static long DispatchCount { get; private set; }

    public static DispatchPath LastPath { get; private set; }

    public static string LastOperation { get; private set; } = string.Empty;

    public static string[] Operations => operations.ToArray();

    public static void Reset()
    {
        DispatchCount = 0;
        LastPath = DispatchPath.None;
        LastOperation = string.Empty;
        operations.Clear();
    }

    internal static void Record(string operation, DispatchPath path)
    {
        DispatchCount++;
        LastPath = path;
        LastOperation = operation;
        operations.Add(operation);
    }
}

internal static class NnDeviceOps
{
    private const int ReductionChunkSize = 256;

    public static Tensor<float> Linear(Tensor<float> input, Tensor<float> weight, Tensor<float> bias, int batch, int inputSize, int outputSize)
    {
        var outputShape = input.Shape.Rank == 2 ? new TensorShape(batch, outputSize) : new TensorShape(outputSize);
        var output = new Tensor<float>(
            outputShape,
            Feather.GPU.CreateBuffer<float>(checked(batch * outputSize)),
            input.RequiresGrad || weight.RequiresGrad || bias.RequiresGrad);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnLinearForwardKernel(
                input.AsReadOnlyBuffer(),
                weight.AsReadOnlyBuffer(),
                bias.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>(inputSize),
                new Uniform<int>(outputSize)),
            output.ElementCount);
        NnDispatchTrace.Record("Linear.Forward", path);
        return output;
    }

    public static Tensor<float> Embedding(Tensor<int> indices, Tensor<float> weight, TensorShape outputShape, int embeddingSize)
    {
        var output = new Tensor<float>(outputShape, Feather.GPU.CreateBuffer<float>(outputShape.ElementCount), weight.RequiresGrad);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnEmbeddingForwardKernel(
                indices.AsReadOnlyBuffer(),
                weight.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>(embeddingSize)),
            output.ElementCount);
        NnDispatchTrace.Record("Embedding.Forward", path);
        return output;
    }

    public static Tensor<float> LayerNorm(Tensor<float> input, Tensor<float> gamma, Tensor<float> beta, int featureSize, float epsilon)
    {
        var output = new Tensor<float>(
            input.Shape,
            Feather.GPU.CreateBuffer<float>(input.ElementCount),
            input.RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnLayerNormForwardKernel(
                input.AsReadOnlyBuffer(),
                gamma.AsReadOnlyBuffer(),
                beta.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>(featureSize),
                new Uniform<float>(epsilon)),
            input.ElementCount / featureSize);
        NnDispatchTrace.Record("LayerNorm.Forward", path);
        return output;
    }

    public static Tensor<float> BatchNorm(Tensor<float> input, Tensor<float> gamma, Tensor<float> beta, Tensor<float> runningMean, Tensor<float> runningVariance, int batch, int featureSize, float epsilon, float momentum, bool training)
    {
        var output = new Tensor<float>(
            input.Shape,
            Feather.GPU.CreateBuffer<float>(input.ElementCount),
            input.RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnBatchNormForwardKernel(
                input.AsReadOnlyBuffer(),
                gamma.AsReadOnlyBuffer(),
                beta.AsReadOnlyBuffer(),
                runningMean.AsReadWriteBuffer(),
                runningVariance.AsReadWriteBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>(batch),
                new Uniform<int>(featureSize),
                new Uniform<float>(epsilon),
                new Uniform<float>(momentum),
                new Uniform<int>(training ? 1 : 0)),
            featureSize);
        NnDispatchTrace.Record("BatchNorm1D.Forward", path);
        return output;
    }

    public static Tensor<float> Activation(Tensor<float> input, NnActivationKind kind)
    {
        if (kind is NnActivationKind.Softmax or NnActivationKind.LogSoftmax)
        {
            return Softmax(input, log: kind == NnActivationKind.LogSoftmax);
        }

        var output = new Tensor<float>(input.Shape, Feather.GPU.CreateBuffer<float>(input.ElementCount), input.RequiresGrad);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnActivationKernel(
                input.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>((int)kind)),
            input.ElementCount);
        NnDispatchTrace.Record($"Activation.{kind}", path);
        return output;
    }

    public static Tensor<float> Softmax(Tensor<float> input, bool log)
    {
        var (rows, classes) = GetMatrixRowsAndColumns(input.Shape, log ? nameof(NnActivationKind.LogSoftmax) : nameof(NnActivationKind.Softmax));
        var output = new Tensor<float>(input.Shape, Feather.GPU.CreateBuffer<float>(input.ElementCount), input.RequiresGrad);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnSoftmaxKernel(
                input.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>(classes),
                new Uniform<int>(log ? 1 : 0)),
            rows);
        NnDispatchTrace.Record(log ? "Activation.LogSoftmax" : "Activation.Softmax", path);
        return output;
    }

    public static Tensor<float> Binary(Tensor<float> left, Tensor<float> right, NnElementwiseBinaryOp op)
    {
        EnsureSameShape(left, right);
        var output = new Tensor<float>(
            left.Shape,
            Feather.GPU.CreateBuffer<float>(left.ElementCount),
            left.RequiresGrad || right.RequiresGrad);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnBinaryElementwiseKernel(
                left.AsReadOnlyBuffer(),
                right.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>((int)op)),
            left.ElementCount);
        NnDispatchTrace.Record($"Elementwise.{op}", path);
        return output;
    }

    public static Tensor<float> Scalar(Tensor<float> input, float scalar, NnElementwiseBinaryOp op)
    {
        var output = new Tensor<float>(input.Shape, Feather.GPU.CreateBuffer<float>(input.ElementCount), input.RequiresGrad);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnScalarElementwiseKernel(
                input.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<float>(scalar),
                new Uniform<int>((int)op)),
            input.ElementCount);
        NnDispatchTrace.Record($"Scalar.{op}", path);
        return output;
    }

    public static void Fill(Tensor<float> tensor, float value)
    {
        var path = Feather.GPU.DispatchAndGetPath(
            new NnFillKernel(tensor.AsReadWriteBuffer(), new Uniform<float>(value)),
            tensor.ElementCount);
        NnDispatchTrace.Record("Fill", path);
    }

    public static void Copy(Tensor<float> source, Tensor<float> destination)
    {
        EnsureSameShape(source, destination);
        var path = Feather.GPU.DispatchAndGetPath(
            new NnCopyKernel(source.AsReadOnlyBuffer(), destination.AsReadWriteBuffer()),
            source.ElementCount);
        NnDispatchTrace.Record("Copy", path);
    }

    public static Tensor<float> MeanSquaredErrorElements(Tensor<float> prediction, Tensor<float> target)
    {
        EnsureSameShape(prediction, target);
        var output = new Tensor<float>(new TensorShape(prediction.ElementCount), Feather.GPU.CreateBuffer<float>(prediction.ElementCount));
        var path = Feather.GPU.DispatchAndGetPath(
            new NnMseElementKernel(
                prediction.AsReadOnlyBuffer(),
                target.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<float>(1f / prediction.ElementCount)),
            prediction.ElementCount);
        NnDispatchTrace.Record("Loss.MseElements", path);
        return output;
    }

    public static Tensor<float> MeanAbsoluteErrorElements(Tensor<float> prediction, Tensor<float> target)
    {
        EnsureSameShape(prediction, target);
        var output = new Tensor<float>(new TensorShape(prediction.ElementCount), Feather.GPU.CreateBuffer<float>(prediction.ElementCount));
        var path = Feather.GPU.DispatchAndGetPath(
            new NnMaeElementKernel(
                prediction.AsReadOnlyBuffer(),
                target.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<float>(1f / prediction.ElementCount)),
            prediction.ElementCount);
        NnDispatchTrace.Record("Loss.MaeElements", path);
        return output;
    }

    public static Tensor<float> CrossEntropyElements(Tensor<float> probabilities, Tensor<int> labels, int batch, int classes)
    {
        var output = new Tensor<float>(new TensorShape(batch), Feather.GPU.CreateBuffer<float>(batch));
        var path = Feather.GPU.DispatchAndGetPath(
            new NnCrossEntropyElementKernel(
                probabilities.AsReadOnlyBuffer(),
                labels.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>(classes),
                new Uniform<float>(1f / batch)),
            batch);
        NnDispatchTrace.Record("Loss.CrossEntropyElements", path);
        return output;
    }

    public static Tensor<float> CrossEntropyFromLogitsElements(Tensor<float> logits, Tensor<int> labels, int batch, int classes)
    {
        var output = new Tensor<float>(new TensorShape(batch), Feather.GPU.CreateBuffer<float>(batch));
        var path = Feather.GPU.DispatchAndGetPath(
            new NnCrossEntropyFromLogitsElementKernel(
                logits.AsReadOnlyBuffer(),
                labels.AsReadOnlyBuffer(),
                output.AsReadWriteBuffer(),
                new Uniform<int>(classes),
                new Uniform<float>(1f / batch)),
            batch);
        NnDispatchTrace.Record("Loss.CrossEntropyFromLogitsElements", path);
        return output;
    }

    public static Tensor<float> Sum(Tensor<float> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.ElementCount <= 0)
        {
            throw new ArgumentException("Reduction input must contain at least one element.", nameof(input));
        }

        var current = input;
        var ownsCurrent = false;
        do
        {
            var partialCount = checked((current.ElementCount + ReductionChunkSize - 1) / ReductionChunkSize);
            var partial = new Tensor<float>(new TensorShape(partialCount), Feather.GPU.CreateBuffer<float>(partialCount));
            var path = Feather.GPU.DispatchAndGetPath(
                new NnPartialSumKernel(
                    current.AsReadOnlyBuffer(),
                    partial.AsReadWriteBuffer(),
                    new Uniform<int>(current.ElementCount),
                    new Uniform<int>(ReductionChunkSize)),
                partialCount);
            NnDispatchTrace.Record("Reduce.PartialSum", path);
            if (ownsCurrent)
            {
                current.Dispose();
            }

            current = partial;
            ownsCurrent = true;
        }
        while (current.ElementCount > 1);

        return current;
    }

    public static void Sgd(Parameter<float> parameter, float learningRate, float weightDecay)
    {
        var path = Feather.GPU.DispatchAndGetPath(
            new NnSgdKernel(
                parameter.Value.AsReadWriteBuffer(),
                parameter.Gradient.AsReadOnlyBuffer(),
                new Uniform<float>(learningRate),
                new Uniform<float>(weightDecay)),
            parameter.ElementCount);
        NnDispatchTrace.Record("Optimizer.SGD", path);
    }

    public static void Momentum(Parameter<float> parameter, Tensor<float> momentum, float learningRate, float momentumFactor, float weightDecay)
    {
        var path = Feather.GPU.DispatchAndGetPath(
            new NnMomentumKernel(
                parameter.Value.AsReadWriteBuffer(),
                parameter.Gradient.AsReadOnlyBuffer(),
                momentum.AsReadWriteBuffer(),
                new Uniform<float>(learningRate),
                new Uniform<float>(momentumFactor),
                new Uniform<float>(weightDecay)),
            parameter.ElementCount);
        NnDispatchTrace.Record("Optimizer.Momentum", path);
    }

    public static void RmsProp(Parameter<float> parameter, Tensor<float> squareAverage, float learningRate, float alpha, float epsilon, float weightDecay)
    {
        var path = Feather.GPU.DispatchAndGetPath(
            new NnRmsPropKernel(
                parameter.Value.AsReadWriteBuffer(),
                parameter.Gradient.AsReadOnlyBuffer(),
                squareAverage.AsReadWriteBuffer(),
                new Uniform<float>(learningRate),
                new Uniform<float>(alpha),
                new Uniform<float>(epsilon),
                new Uniform<float>(weightDecay)),
            parameter.ElementCount);
        NnDispatchTrace.Record("Optimizer.RMSProp", path);
    }

    public static void Adam(Parameter<float> parameter, Tensor<float> firstMoment, Tensor<float> secondMoment, float learningRate, float beta1, float beta2, float epsilon, int step, float weightDecay, float gradientClip, bool decoupledWeightDecay)
    {
        var path = Feather.GPU.DispatchAndGetPath(
            new NnAdamKernel(
                parameter.Value.AsReadWriteBuffer(),
                parameter.Gradient.AsReadOnlyBuffer(),
                firstMoment.AsReadWriteBuffer(),
                secondMoment.AsReadWriteBuffer(),
                new Uniform<float>(learningRate),
                new Uniform<float>(beta1),
                new Uniform<float>(beta2),
                new Uniform<float>(epsilon),
                new Uniform<int>(step),
                new Uniform<float>(weightDecay),
                new Uniform<float>(gradientClip),
                new Uniform<int>(decoupledWeightDecay ? 1 : 0)),
            parameter.ElementCount);
        NnDispatchTrace.Record(decoupledWeightDecay ? "Optimizer.AdamW" : "Optimizer.Adam", path);
    }

    private static void EnsureSameShape(Tensor<float> left, Tensor<float> right)
    {
        if (!left.Shape.Equals(right.Shape))
        {
            throw new ArgumentException("Tensor shapes must match.");
        }
    }

    private static (int Rows, int Columns) GetMatrixRowsAndColumns(TensorShape shape, string operation)
    {
        if (shape.Rank != 2)
        {
            throw new ArgumentException($"{operation} expects a rank-2 [batch, classes] tensor.");
        }

        var rows = shape[0];
        var columns = shape[1];
        if (rows <= 0 || columns <= 0)
        {
            throw new ArgumentException($"{operation} tensor dimensions must be positive.");
        }

        return (rows, columns);
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnLinearForwardKernel(
    ReadOnlyBuffer<float> input,
    ReadOnlyBuffer<float> weight,
    ReadOnlyBuffer<float> bias,
    ReadWriteBuffer<float> output,
    Uniform<int> inputSize,
    Uniform<int> outputSize) : IKernel1D
{
    public void Execute()
    {
        int outputIndex = ThreadIds.X;
        int feature = outputIndex % outputSize.Value;
        int row = outputIndex / outputSize.Value;
        int inputOffset = row * inputSize.Value;
        int weightOffset = feature * inputSize.Value;
        float sum = bias[feature];
        for (int i = 0; i < inputSize.Value; i++)
        {
            sum += input[inputOffset + i] * weight[weightOffset + i];
        }

        output[outputIndex] = sum;
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnActivationKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output,
    Uniform<int> kind) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float value = input[i];
        if (kind.Value == 0)
        {
            output[i] = value > 0f ? value : 0f;
        }
        else if (kind.Value == 1)
        {
            output[i] = 1f / (1f + ShaderMath.Exp(-value));
        }
        else if (kind.Value == 2)
        {
            float e = ShaderMath.Exp(2f * value);
            output[i] = (e - 1f) / (e + 1f);
        }
        else
        {
            output[i] = value / (1f + ShaderMath.Exp(-value));
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnSoftmaxKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output,
    Uniform<int> classes,
    Uniform<int> log) : IKernel1D
{
    public void Execute()
    {
        int row = ThreadIds.X;
        int offset = row * classes.Value;
        float maximum = input[offset];
        for (int cls = 1; cls < classes.Value; cls++)
        {
            float value = input[offset + cls];
            if (value > maximum)
            {
                maximum = value;
            }
        }

        float sumExp = 0f;
        for (int cls = 0; cls < classes.Value; cls++)
        {
            sumExp += ShaderMath.Exp(input[offset + cls] - maximum);
        }

        float logSum = ShaderMath.Log(sumExp);
        for (int cls = 0; cls < classes.Value; cls++)
        {
            float shifted = input[offset + cls] - maximum;
            if (log.Value != 0)
            {
                output[offset + cls] = shifted - logSum;
            }
            else
            {
                output[offset + cls] = ShaderMath.Exp(shifted) / sumExp;
            }
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnEmbeddingForwardKernel(
    ReadOnlyBuffer<int> indices,
    ReadOnlyBuffer<float> weight,
    ReadWriteBuffer<float> output,
    Uniform<int> embeddingSize) : IKernel1D
{
    public void Execute()
    {
        int outputIndex = ThreadIds.X;
        int feature = outputIndex % embeddingSize.Value;
        int row = outputIndex / embeddingSize.Value;
        int index = indices[row];
        output[outputIndex] = weight[(index * embeddingSize.Value) + feature];
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnLayerNormForwardKernel(
    ReadOnlyBuffer<float> input,
    ReadOnlyBuffer<float> gamma,
    ReadOnlyBuffer<float> beta,
    ReadWriteBuffer<float> output,
    Uniform<int> featureSize,
    Uniform<float> epsilon) : IKernel1D
{
    public void Execute()
    {
        int row = ThreadIds.X;
        int offset = row * featureSize.Value;
        float mean = 0f;
        for (int feature = 0; feature < featureSize.Value; feature++)
        {
            mean += input[offset + feature];
        }

        mean /= featureSize.Value;
        float variance = 0f;
        for (int feature = 0; feature < featureSize.Value; feature++)
        {
            float centered = input[offset + feature] - mean;
            variance += centered * centered;
        }

        variance /= featureSize.Value;
        float inverseStdDev = 1f / ShaderMath.Sqrt(variance + epsilon.Value);
        for (int feature = 0; feature < featureSize.Value; feature++)
        {
            float normalized = (input[offset + feature] - mean) * inverseStdDev;
            output[offset + feature] = (normalized * gamma[feature]) + beta[feature];
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnBatchNormForwardKernel(
    ReadOnlyBuffer<float> input,
    ReadOnlyBuffer<float> gamma,
    ReadOnlyBuffer<float> beta,
    ReadWriteBuffer<float> runningMean,
    ReadWriteBuffer<float> runningVariance,
    ReadWriteBuffer<float> output,
    Uniform<int> batch,
    Uniform<int> featureSize,
    Uniform<float> epsilon,
    Uniform<float> momentum,
    Uniform<int> training) : IKernel1D
{
    public void Execute()
    {
        int feature = ThreadIds.X;
        float mean = runningMean[feature];
        float variance = runningVariance[feature];
        if (training.Value != 0)
        {
            mean = 0f;
            for (int row = 0; row < batch.Value; row++)
            {
                mean += input[(row * featureSize.Value) + feature];
            }

            mean /= batch.Value;
            variance = 0f;
            for (int row = 0; row < batch.Value; row++)
            {
                float centered = input[(row * featureSize.Value) + feature] - mean;
                variance += centered * centered;
            }

            variance /= batch.Value;
            runningMean[feature] = ((1f - momentum.Value) * runningMean[feature]) + (momentum.Value * mean);
            runningVariance[feature] = ((1f - momentum.Value) * runningVariance[feature]) + (momentum.Value * variance);
        }

        float inverseStdDev = 1f / ShaderMath.Sqrt(variance + epsilon.Value);
        for (int row = 0; row < batch.Value; row++)
        {
            int index = (row * featureSize.Value) + feature;
            float normalized = (input[index] - mean) * inverseStdDev;
            output[index] = (normalized * gamma[feature]) + beta[feature];
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnBinaryElementwiseKernel(
    ReadOnlyBuffer<float> left,
    ReadOnlyBuffer<float> right,
    ReadWriteBuffer<float> output,
    Uniform<int> op) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float l = left[i];
        float r = right[i];
        if (op.Value == 0)
        {
            output[i] = l + r;
        }
        else if (op.Value == 1)
        {
            output[i] = l - r;
        }
        else if (op.Value == 2)
        {
            output[i] = l * r;
        }
        else
        {
            output[i] = l / r;
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnScalarElementwiseKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output,
    Uniform<float> scalar,
    Uniform<int> op) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float value = input[i];
        if (op.Value == 0)
        {
            output[i] = value + scalar.Value;
        }
        else if (op.Value == 1)
        {
            output[i] = value - scalar.Value;
        }
        else if (op.Value == 2)
        {
            output[i] = value * scalar.Value;
        }
        else
        {
            output[i] = value / scalar.Value;
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnFillKernel(ReadWriteBuffer<float> output, Uniform<float> value) : IKernel1D
{
    public void Execute()
    {
        output[ThreadIds.X] = value.Value;
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnCopyKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i];
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnMseElementKernel(
    ReadOnlyBuffer<float> prediction,
    ReadOnlyBuffer<float> target,
    ReadWriteBuffer<float> output,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float diff = prediction[i] - target[i];
        output[i] = diff * diff * scale.Value;
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnMaeElementKernel(
    ReadOnlyBuffer<float> prediction,
    ReadOnlyBuffer<float> target,
    ReadWriteBuffer<float> output,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float diff = prediction[i] - target[i];
        output[i] = (diff < 0f ? -diff : diff) * scale.Value;
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnCrossEntropyElementKernel(
    ReadOnlyBuffer<float> probabilities,
    ReadOnlyBuffer<int> labels,
    ReadWriteBuffer<float> output,
    Uniform<int> classes,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int row = ThreadIds.X;
        int label = labels[row];
        if (label < 0 || label >= classes.Value)
        {
            output[row] = ShaderMath.Log(-1f);
        }
        else
        {
            float probability = probabilities[(row * classes.Value) + label];
            if (probability < 0.0000001f)
            {
                probability = 0.0000001f;
            }

            output[row] = -ShaderMath.Log(probability) * scale.Value;
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnCrossEntropyFromLogitsElementKernel(
    ReadOnlyBuffer<float> logits,
    ReadOnlyBuffer<int> labels,
    ReadWriteBuffer<float> output,
    Uniform<int> classes,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int row = ThreadIds.X;
        int offset = row * classes.Value;
        int label = labels[row];
        if (label < 0 || label >= classes.Value)
        {
            output[row] = ShaderMath.Log(-1f);
        }
        else
        {
            float maximum = logits[offset];
            for (int cls = 1; cls < classes.Value; cls++)
            {
                float value = logits[offset + cls];
                if (value > maximum)
                {
                    maximum = value;
                }
            }

            float sumExp = 0f;
            for (int cls = 0; cls < classes.Value; cls++)
            {
                sumExp += ShaderMath.Exp(logits[offset + cls] - maximum);
            }

            output[row] = (ShaderMath.Log(sumExp) + maximum - logits[offset + label]) * scale.Value;
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnPartialSumKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output,
    Uniform<int> count,
    Uniform<int> chunkSize) : IKernel1D
{
    public void Execute()
    {
        int chunk = ThreadIds.X;
        int start = chunk * chunkSize.Value;
        int end = start + chunkSize.Value;
        if (end > count.Value)
        {
            end = count.Value;
        }

        float sum = 0f;
        for (int i = start; i < end; i++)
        {
            sum += input[i];
        }

        output[chunk] = sum;
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnSgdKernel(
    ReadWriteBuffer<float> value,
    ReadOnlyBuffer<float> gradient,
    Uniform<float> learningRate,
    Uniform<float> weightDecay) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float current = value[i];
        float grad = gradient[i] + (weightDecay.Value * current);
        value[i] = current - (learningRate.Value * grad);
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnMomentumKernel(
    ReadWriteBuffer<float> value,
    ReadOnlyBuffer<float> gradient,
    ReadWriteBuffer<float> momentum,
    Uniform<float> learningRate,
    Uniform<float> momentumFactor,
    Uniform<float> weightDecay) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float current = value[i];
        float grad = gradient[i] + (weightDecay.Value * current);
        float nextMomentum = (momentumFactor.Value * momentum[i]) + grad;
        momentum[i] = nextMomentum;
        value[i] = current - (learningRate.Value * nextMomentum);
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnRmsPropKernel(
    ReadWriteBuffer<float> value,
    ReadOnlyBuffer<float> gradient,
    ReadWriteBuffer<float> squareAverage,
    Uniform<float> learningRate,
    Uniform<float> alpha,
    Uniform<float> epsilon,
    Uniform<float> weightDecay) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float current = value[i];
        float grad = gradient[i] + (weightDecay.Value * current);
        float avg = (alpha.Value * squareAverage[i]) + ((1f - alpha.Value) * grad * grad);
        squareAverage[i] = avg;
        value[i] = current - (learningRate.Value * grad / (ShaderMath.Sqrt(avg) + epsilon.Value));
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct NnAdamKernel(
    ReadWriteBuffer<float> value,
    ReadOnlyBuffer<float> gradient,
    ReadWriteBuffer<float> firstMoment,
    ReadWriteBuffer<float> secondMoment,
    Uniform<float> learningRate,
    Uniform<float> beta1,
    Uniform<float> beta2,
    Uniform<float> epsilon,
    Uniform<int> step,
    Uniform<float> weightDecay,
    Uniform<float> gradientClip,
    Uniform<int> decoupledWeightDecay) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float current = value[i];
        float grad = gradient[i];
        if (decoupledWeightDecay.Value == 0)
        {
            grad += weightDecay.Value * current;
        }

        if (gradientClip.Value > 0f)
        {
            grad = ShaderMath.Clamp(grad, -gradientClip.Value, gradientClip.Value);
        }

        float m = (beta1.Value * firstMoment[i]) + ((1f - beta1.Value) * grad);
        float v = (beta2.Value * secondMoment[i]) + ((1f - beta2.Value) * grad * grad);
        firstMoment[i] = m;
        secondMoment[i] = v;

        float correctedM = m / (1f - ShaderMath.Pow(beta1.Value, step.Value));
        float correctedV = v / (1f - ShaderMath.Pow(beta2.Value, step.Value));
        float next = current - (learningRate.Value * correctedM / (ShaderMath.Sqrt(correctedV) + epsilon.Value));
        if (decoupledWeightDecay.Value != 0)
        {
            next -= learningRate.Value * weightDecay.Value * current;
        }

        value[i] = next;
    }
}
