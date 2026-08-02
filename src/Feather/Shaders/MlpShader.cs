// Injected into consuming projects as SOURCE by FeatherCompute.targets. See FeatherCamera.cs for
// why a shared shader helper cannot ship as a compiled assembly.

namespace Feather.Shaders;

/// <summary>
/// Packed MLP weight-layout arithmetic for shader code.
/// </summary>
/// <remarks>
/// The offsets a shader needs to index a network trained by
/// <c>Feather.NN.MlpRegression3To1LossKernel</c>, mirroring <c>Feather.NN.MlpLayout</c> so a hand-written
/// inference pass does not keep its own copy of the layout in sync with the trainer's by hand.
///
/// It is only the arithmetic, not the evaluation, and that is a platform limitation rather than a choice.
/// A <c>[Callable]</c> cannot take a buffer parameter: the generator's <c>IsSupportedCallableType</c>
/// accepts shader resource views, but <c>GPU::IR::Type</c> has no buffer kind, so
/// <c>GPU::IR::CallableParameter</c> cannot represent one and the native typed-IR lowerer fails with
/// "section 7 typed IR lowerer failed before EasyGPU module creation" at dispatch rather than at compile
/// time. <c>MlpLoweringBoundaryTests</c> pins that behavior. Until the IR gains a resource type, a shared
/// helper that reads weights out of a buffer is not expressible.
///
/// So there are two routes to inference, and neither is this class:
///
/// <list type="bullet">
/// <item>Dispatch <c>Feather.NN.MlpInference3To1Kernel</c>. It is an ordinary compiled kernel, so it
/// indexes its buffers directly. This is the route for evaluating a batch of inputs.</item>
/// <item>Write the evaluation inline in your own kernel or fragment shader, using the offsets below.
/// This is the route for a render pass that evaluates per pixel and never wants a round trip through a
/// buffer.</item>
/// </list>
///
/// Layout, in order: <c>w1[h,3]</c>, <c>b1[h]</c>, <c>w2[h,h]</c>, <c>b2[h]</c>, <c>w3[1,h]</c>,
/// <c>b3[1]</c>, row-major per layer as <c>w[outIndex * inputSize + inIndex]</c>.
/// </remarks>
[ShaderLibrary]
public static class MlpShader
{
    /// <summary>
    /// Flat element count for a packed 3→h→h→1 network, for buffer sizing.
    /// </summary>
    /// <remarks>Mirrors <c>Feather.NN.MlpLayout.PackedElementCount3To1</c>.</remarks>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    [Callable]
    public static int PackedElementCount3To1(int hiddenSize)
    {
        return (hiddenSize * hiddenSize) + (6 * hiddenSize) + 1;
    }

    /// <summary>
    /// Scratch elements one lane needs to hold a 3→h→h→1 network's activations.
    /// </summary>
    /// <remarks>
    /// Mirrors <c>Feather.NN.MlpLayout.ScratchElementsPerLane3To1</c>, so one scratch buffer serves both
    /// training and inference. A lane's base is <c>laneIndex * ScratchElementsPerLane3To1(hiddenSize)</c>;
    /// lanes that share a range corrupt each other's activations.
    ///
    /// Within a lane, <c>3 + 2 * hiddenSize</c> elements are enough for inference — three staged inputs
    /// and two activation vectors — which fits this stride for any hidden size of 2 or more.
    /// </remarks>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    [Callable]
    public static int ScratchElementsPerLane3To1(int hiddenSize)
    {
        return 4 * hiddenSize;
    }

    /// <summary>Offset of the first hidden layer's weight matrix, shaped <c>[hiddenSize, 3]</c>.</summary>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    [Callable]
    public static int Layer1WeightOffset3To1(int hiddenSize)
    {
        return 0;
    }

    /// <summary>Offset of the first hidden layer's bias vector, shaped <c>[hiddenSize]</c>.</summary>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    [Callable]
    public static int Layer1BiasOffset3To1(int hiddenSize)
    {
        return 3 * hiddenSize;
    }

    /// <summary>Offset of the second hidden layer's weight matrix, shaped <c>[hiddenSize, hiddenSize]</c>.</summary>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    [Callable]
    public static int Layer2WeightOffset3To1(int hiddenSize)
    {
        return 4 * hiddenSize;
    }

    /// <summary>Offset of the second hidden layer's bias vector, shaped <c>[hiddenSize]</c>.</summary>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    [Callable]
    public static int Layer2BiasOffset3To1(int hiddenSize)
    {
        return (4 * hiddenSize) + (hiddenSize * hiddenSize);
    }

    /// <summary>Offset of the output layer's weight row, shaped <c>[1, hiddenSize]</c>.</summary>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    [Callable]
    public static int OutputWeightOffset3To1(int hiddenSize)
    {
        return (5 * hiddenSize) + (hiddenSize * hiddenSize);
    }

    /// <summary>Offset of the output layer's single bias value.</summary>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    [Callable]
    public static int OutputBiasOffset3To1(int hiddenSize)
    {
        return (6 * hiddenSize) + (hiddenSize * hiddenSize);
    }

    /// <summary>Index of a weight within a row-major layer matrix, relative to the layer's offset.</summary>
    /// <remarks>
    /// Present so an inline evaluation reads <c>weights[w2 + MlpShader.WeightIndex(j, k, hiddenSize)]</c>
    /// rather than open-coding the row stride, which is the detail easiest to get wrong when a hidden
    /// layer is not square.
    /// </remarks>
    /// <param name="outputIndex">The output neuron.</param>
    /// <param name="inputIndex">The input the weight applies to.</param>
    /// <param name="inputSize">The number of inputs to the layer.</param>
    [Callable]
    public static int WeightIndex(int outputIndex, int inputIndex, int inputSize)
    {
        return (outputIndex * inputSize) + inputIndex;
    }
}
