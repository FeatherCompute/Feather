namespace Feather.NN;

/// <summary>
/// Base type for elementwise activation modules.
/// </summary>
public abstract class Activation : Module
{
    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => [];

    /// <summary>
    /// Applies the activation to a tensor.
    /// </summary>
    public Tensor<float> Forward(Tensor<float> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        return NnDeviceOps.Activation(input, Kind);
    }

    /// <summary>
    /// Gets the device activation opcode.
    /// </summary>
    protected abstract NnActivationKind Kind { get; }
}

/// <summary>
/// Rectified linear unit activation.
/// </summary>
public sealed class ReLU : Activation
{
    /// <inheritdoc />
    protected override NnActivationKind Kind => NnActivationKind.ReLU;
}

/// <summary>
/// Sigmoid activation.
/// </summary>
public sealed class Sigmoid : Activation
{
    /// <inheritdoc />
    protected override NnActivationKind Kind => NnActivationKind.Sigmoid;
}

/// <summary>
/// Hyperbolic tangent activation.
/// </summary>
public sealed class Tanh : Activation
{
    /// <inheritdoc />
    protected override NnActivationKind Kind => NnActivationKind.Tanh;
}

/// <summary>
/// Sigmoid linear unit activation.
/// </summary>
public sealed class SiLU : Activation
{
    /// <inheritdoc />
    protected override NnActivationKind Kind => NnActivationKind.SiLU;
}

/// <summary>
/// Row-wise softmax activation for rank-2 logits.
/// </summary>
public sealed class Softmax : Activation
{
    /// <inheritdoc />
    protected override NnActivationKind Kind => NnActivationKind.Softmax;
}

/// <summary>
/// Row-wise log-softmax activation for rank-2 logits.
/// </summary>
public sealed class LogSoftmax : Activation
{
    /// <inheritdoc />
    protected override NnActivationKind Kind => NnActivationKind.LogSoftmax;
}

/// <summary>
/// Device-backed tensor operations for float tensors.
/// </summary>
public static class TensorOps
{
    /// <summary>
    /// Adds two same-shaped tensors on the device.
    /// </summary>
    public static Tensor<float> Add(Tensor<float> left, Tensor<float> right)
        => NnDeviceOps.Binary(RequireTensor(left), RequireTensor(right), NnElementwiseBinaryOp.Add);

    /// <summary>
    /// Subtracts <paramref name="right"/> from <paramref name="left"/> on the device.
    /// </summary>
    public static Tensor<float> Subtract(Tensor<float> left, Tensor<float> right)
        => NnDeviceOps.Binary(RequireTensor(left), RequireTensor(right), NnElementwiseBinaryOp.Subtract);

    /// <summary>
    /// Multiplies two same-shaped tensors on the device.
    /// </summary>
    public static Tensor<float> Multiply(Tensor<float> left, Tensor<float> right)
        => NnDeviceOps.Binary(RequireTensor(left), RequireTensor(right), NnElementwiseBinaryOp.Multiply);

    /// <summary>
    /// Divides <paramref name="left"/> by <paramref name="right"/> on the device.
    /// </summary>
    public static Tensor<float> Divide(Tensor<float> left, Tensor<float> right)
        => NnDeviceOps.Binary(RequireTensor(left), RequireTensor(right), NnElementwiseBinaryOp.Divide);

    /// <summary>
    /// Adds a scalar to each tensor element on the device.
    /// </summary>
    public static Tensor<float> Add(Tensor<float> input, float scalar)
        => NnDeviceOps.Scalar(RequireTensor(input), scalar, NnElementwiseBinaryOp.Add);

    /// <summary>
    /// Subtracts a scalar from each tensor element on the device.
    /// </summary>
    public static Tensor<float> Subtract(Tensor<float> input, float scalar)
        => NnDeviceOps.Scalar(RequireTensor(input), scalar, NnElementwiseBinaryOp.Subtract);

    /// <summary>
    /// Multiplies each tensor element by a scalar on the device.
    /// </summary>
    public static Tensor<float> Multiply(Tensor<float> input, float scalar)
        => NnDeviceOps.Scalar(RequireTensor(input), scalar, NnElementwiseBinaryOp.Multiply);

    /// <summary>
    /// Divides each tensor element by a scalar on the device.
    /// </summary>
    public static Tensor<float> Divide(Tensor<float> input, float scalar)
        => NnDeviceOps.Scalar(RequireTensor(input), scalar, NnElementwiseBinaryOp.Divide);

    /// <summary>
    /// Copies tensor values on the device.
    /// </summary>
    public static void Copy(Tensor<float> source, Tensor<float> destination)
        => NnDeviceOps.Copy(RequireTensor(source), RequireTensor(destination));

    /// <summary>
    /// Fills a tensor on the device.
    /// </summary>
    public static void Fill(Tensor<float> tensor, float value)
        => NnDeviceOps.Fill(RequireTensor(tensor), value);

    /// <summary>
    /// Applies row-wise softmax to rank-2 logits on the device.
    /// </summary>
    public static Tensor<float> Softmax(Tensor<float> logits)
        => NnDeviceOps.Softmax(RequireTensor(logits), log: false);

    /// <summary>
    /// Applies row-wise log-softmax to rank-2 logits on the device.
    /// </summary>
    public static Tensor<float> LogSoftmax(Tensor<float> logits)
        => NnDeviceOps.Softmax(RequireTensor(logits), log: true);

    private static Tensor<float> RequireTensor(Tensor<float> tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        return tensor;
    }
}

/// <summary>
/// Device-backed loss functions for Feather tensors.
/// </summary>
public static class Losses
{
    /// <summary>
    /// Computes per-element scaled mean squared error contributions on the device.
    /// </summary>
    public static Tensor<float> MeanSquaredErrorTensor(Tensor<float> prediction, Tensor<float> target)
    {
        using var elements = NnDeviceOps.MeanSquaredErrorElements(prediction, target);
        return NnDeviceOps.Sum(elements);
    }

    /// <summary>
    /// Computes the mean squared error between prediction and target tensors and reads the scalar value.
    /// </summary>
    public static float MeanSquaredError(Tensor<float> prediction, Tensor<float> target)
    {
        using var loss = MeanSquaredErrorTensor(prediction, target);
        return loss.Buffer.ToArray()[0];
    }

    /// <summary>
    /// Computes per-element scaled mean absolute error contributions on the device.
    /// </summary>
    public static Tensor<float> MeanAbsoluteErrorTensor(Tensor<float> prediction, Tensor<float> target)
    {
        using var elements = NnDeviceOps.MeanAbsoluteErrorElements(prediction, target);
        return NnDeviceOps.Sum(elements);
    }

    /// <summary>
    /// Computes mean absolute error and reads the scalar value.
    /// </summary>
    public static float MeanAbsoluteError(Tensor<float> prediction, Tensor<float> target)
    {
        using var loss = MeanAbsoluteErrorTensor(prediction, target);
        return loss.Buffer.ToArray()[0];
    }

    /// <summary>
    /// Computes average cross entropy from probabilities and device integer class labels.
    /// </summary>
    public static Tensor<float> CrossEntropyTensor(Tensor<float> probabilities, Tensor<int> labels)
    {
        var (batch, classes) = ValidateClassificationShapes(probabilities, labels, nameof(probabilities), nameof(labels), "CrossEntropy");

        using var elements = NnDeviceOps.CrossEntropyElements(probabilities, labels, batch, classes);
        return NnDeviceOps.Sum(elements);
    }

    /// <summary>
    /// Computes average cross entropy from probabilities and device integer class labels, then reads the scalar value.
    /// </summary>
    public static float CrossEntropy(Tensor<float> probabilities, Tensor<int> labels)
    {
        using var loss = CrossEntropyTensor(probabilities, labels);
        return loss.Buffer.ToArray()[0];
    }

    /// <summary>
    /// Computes average softmax cross entropy directly from logits and device integer class labels.
    /// </summary>
    public static Tensor<float> CrossEntropyFromLogitsTensor(Tensor<float> logits, Tensor<int> labels)
    {
        var (batch, classes) = ValidateClassificationShapes(logits, labels, nameof(logits), nameof(labels), "CrossEntropyFromLogits");

        using var elements = NnDeviceOps.CrossEntropyFromLogitsElements(logits, labels, batch, classes);
        return NnDeviceOps.Sum(elements);
    }

    /// <summary>
    /// Computes average softmax cross entropy from logits and device integer class labels, then reads the scalar value.
    /// </summary>
    public static float CrossEntropyFromLogits(Tensor<float> logits, Tensor<int> labels)
    {
        using var loss = CrossEntropyFromLogitsTensor(logits, labels);
        return loss.Buffer.ToArray()[0];
    }

    /// <summary>
    /// Computes average cross entropy from probabilities and host integer class labels.
    /// </summary>
    public static float CrossEntropy(Tensor<float> probabilities, ReadOnlySpan<int> labels)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        ValidateProbabilityShape(probabilities, nameof(probabilities), "CrossEntropy", out var batch, out var classes);

        if (labels.Length != batch)
        {
            throw new ArgumentException("Label count must match the batch dimension.", nameof(labels));
        }

        // Host-label overloads are explicit readback conveniences. Use CrossEntropyTensor with a label tensor
        // for the device path.
        var values = probabilities.Buffer.ToArray();
        var sum = 0f;
        for (var row = 0; row < batch; row++)
        {
            var label = labels[row];
            if ((uint)label >= (uint)classes)
            {
                throw new ArgumentOutOfRangeException(nameof(labels), "Label is outside the class range.");
            }

            var probability = MathF.Max(values[(row * classes) + label], 1e-7f);
            sum -= MathF.Log(probability);
        }

        return sum / batch;
    }

    /// <summary>
    /// Computes average softmax cross entropy from logits and host integer class labels.
    /// </summary>
    public static float CrossEntropyFromLogits(Tensor<float> logits, ReadOnlySpan<int> labels)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ValidateProbabilityShape(logits, nameof(logits), "CrossEntropyFromLogits", out var batch, out var classes);
        if (labels.Length != batch)
        {
            throw new ArgumentException("Label count must match the batch dimension.", nameof(labels));
        }

        // Host-label overloads are explicit readback conveniences. Use CrossEntropyFromLogitsTensor with
        // a label tensor for the device path.
        var values = logits.Buffer.ToArray();
        var sum = 0f;
        for (var row = 0; row < batch; row++)
        {
            var label = labels[row];
            if ((uint)label >= (uint)classes)
            {
                throw new ArgumentOutOfRangeException(nameof(labels), "Label is outside the class range.");
            }

            var offset = row * classes;
            var maximum = values[offset];
            for (var cls = 1; cls < classes; cls++)
            {
                maximum = MathF.Max(maximum, values[offset + cls]);
            }

            var sumExp = 0f;
            for (var cls = 0; cls < classes; cls++)
            {
                sumExp += MathF.Exp(values[offset + cls] - maximum);
            }

            sum += MathF.Log(sumExp) + maximum - values[offset + label];
        }

        return sum / batch;
    }

    private static (int Batch, int Classes) ValidateClassificationShapes(
        Tensor<float> values,
        Tensor<int> labels,
        string valuesParamName,
        string labelsParamName,
        string operation)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(labels);
        ValidateProbabilityShape(values, valuesParamName, operation, out var batch, out var classes);
        if (labels.Shape.Rank != 1 || labels.Shape.Dimensions[0] != batch)
        {
            throw new ArgumentException("Label tensor must be rank-1 with length matching the batch dimension.", labelsParamName);
        }

        return (batch, classes);
    }

    private static void ValidateProbabilityShape(Tensor<float> values, string paramName, string operation, out int batch, out int classes)
    {
        if (values.Shape.Rank != 2)
        {
            throw new ArgumentException($"{operation} expects a rank-2 [batch, classes] tensor.", paramName);
        }

        batch = values.Shape.Dimensions[0];
        classes = values.Shape.Dimensions[1];
    }
}

/// <summary>
/// Module-style cross-entropy loss facade for high-level NN workflows.
/// </summary>
public sealed class CrossEntropyLoss
{
    /// <summary>
    /// Computes average cross entropy from probabilities and device integer class labels.
    /// </summary>
    public Tensor<float> Forward(Tensor<float> probabilities, Tensor<int> labels)
        => Losses.CrossEntropyTensor(probabilities, labels);

    /// <summary>
    /// Computes average cross entropy from probabilities and host integer class labels.
    /// </summary>
    public float Forward(Tensor<float> probabilities, ReadOnlySpan<int> labels)
        => Losses.CrossEntropy(probabilities, labels);

    /// <summary>
    /// Computes average softmax cross entropy directly from logits and device integer class labels.
    /// </summary>
    public Tensor<float> FromLogits(Tensor<float> logits, Tensor<int> labels)
        => Losses.CrossEntropyFromLogitsTensor(logits, labels);

    /// <summary>
    /// Computes average softmax cross entropy directly from logits and host integer class labels.
    /// </summary>
    public float FromLogits(Tensor<float> logits, ReadOnlySpan<int> labels)
        => Losses.CrossEntropyFromLogits(logits, labels);
}
