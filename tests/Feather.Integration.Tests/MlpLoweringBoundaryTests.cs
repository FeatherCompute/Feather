using Feather;
using Feather.Interop;
using Feather.Native;
using Feather.Resources;

namespace Feather.Integration.Tests;

/// <summary>
/// Pins the <c>[Callable]</c> resource-parameter boundary that shapes how MLP inference is exposed.
/// </summary>
/// <remarks>
/// Buffer parameters lower by specializing the callable's resource access to the bound SSBO.
/// <c>MlpShader</c> still exposes layout arithmetic and inference still ships as
/// <see cref="Feather.NN.MlpInference3To1Kernel" /> until that API is deliberately consolidated.
/// </remarks>
public class MlpLoweringBoundaryTests
{
}
