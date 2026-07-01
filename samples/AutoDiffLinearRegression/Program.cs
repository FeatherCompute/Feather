using Feather;
using Feather.AD;
using Feather.Math;
using Feather.Resources;

// Linear regression: y = W * x + b.
// This sample demonstrates the current [AutoDiff] authoring surface without claiming completed training validation.

const int N = 64;
float[] x = new float[N];
float[] y = new float[N];
var rng = new Random(42);
for (int i = 0; i < N; i++)
{
    float xi = (float)i / N;
    x[i] = xi;
    y[i] = 2.0f * xi + 1.0f + (float)(rng.NextDouble() - 0.5) * 0.1f;
}

using var bufX = GPU.CreateBuffer<float>(x, BufferAccess.ReadOnly);
using var bufY = GPU.CreateBuffer<float>(y, BufferAccess.ReadOnly);
using var bufW = GPU.CreateBuffer<float>([0.5f], BufferAccess.ReadWrite);
using var bufB = GPU.CreateBuffer<float>([0.0f], BufferAccess.ReadWrite);

Console.WriteLine("AD Linear Regression (conceptual - uses [AutoDiff] attribute)");
Console.WriteLine("The kernel builder activates the GradientTape when AutoDiff is true.");
Console.WriteLine("Backward-pass validation is outside the completed basic DSL proof set, so this sample does not print PASS.");

GPU.Dispatch(new LinearKernel(bufX.AsReadOnly(), bufY.AsReadOnly(), bufW.AsReadWrite(), bufB.AsReadWrite()), N);

float[] wResult = bufW.ToArray();
float[] bResult = bufB.ToArray();
Console.WriteLine($"W buffer = {wResult[0]:F4}, b buffer = {bResult[0]:F4}");

/// <summary>
/// Linear regression kernel with AutoDiff annotations.
/// </summary>
[Kernel]
[AutoDiff]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct LinearKernel(
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> w,
    ReadWriteBuffer<float> b) : IKernel1D
{
    /// <summary>
    /// Computes a per-element squared loss and records conceptual AD annotations.
    /// </summary>
    public void Execute()
    {
        int i = ThreadIds.X;
        float xi = x[i];
        float yi = y[i];
        float wVal = w[0];
        float bVal = b[0];

        float pred = wVal * xi + bVal;
        float diff = pred - yi;
        float loss = diff * diff;

        AD.Parameter(w[0]);
        AD.Parameter(b[0]);
        AD.Loss(loss);

        w[0] = loss;
    }
}
