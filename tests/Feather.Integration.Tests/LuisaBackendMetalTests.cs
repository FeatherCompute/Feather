using System.Diagnostics;

namespace Feather.Integration.Tests;

public class LuisaBackendMetalTests
{
    private const string Prefix = "Feather.Integration.Tests.";

    public static IEnumerable<object?[]> ParityCases =>
    [
        Case("LuisaBackendAdTests.ReverseModeGradientsMatchEasyGpu"),
        Case("LuisaBackendAdTests.VectorParameterGradientsMatchEasyGpu"),
        Case("LuisaBackendAdTests.CallableGradientsExecuteThroughLuisa"),
        Case("LuisaBackendAdTests.ConditionalIntrinsicGradientsMatchEasyGpu"),
        Case("LuisaBackendControlMemoryTests.StructuredControlFlowAndMutableLocalsMatchEasyGpu"),
        Case("LuisaBackendControlMemoryTests.SharedMemoryLocalIdsAndBarrierMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.FullIntegerAtomicMatrixMatchesEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.TwoAndThreeDimensionalDispatchIdsMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.Texture2DAndTexture3DLoadStoreMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.TextureSamplingAndMixedResourceOrderMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.TextureSampleGradExecutesThroughLuisaXir"),
        Case("LuisaBackendResourceDispatchTests.NestedScalarCallablesMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.ShaderLibraryBufferCallablesMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.ShaderLibraryTextureAndSamplerCallablesMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.MutableGpuStructCallablesAndWritebackMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.StructArraysAndNestedWritebackMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.NonDivisibleLogicalBoundsMatchEasyGpu"),
        Case("LuisaBackendTests.VectorAddExecutesThroughLuisaXirVulkan"),
        Case("LuisaBackendTypeFeatureTests.ScalarsComparisonsLogicAndBitOperationsMatchEasyGpu"),
        Case("LuisaBackendTypeFeatureTests.UnsignedConstantsBitOperationsComparisonsAndConversionsMatchEasyGpu"),
        Case("LuisaBackendTypeFeatureTests.VectorConstructionAndSwizzlesMatchEasyGpu"),
        Case("LuisaBackendTypeFeatureTests.MatrixConstructionAndLinearAlgebraMatchEasyGpu"),
        Case("LuisaBackendTypeFeatureTests.StructAggregateLoadsAndNestedFieldExtractionMatchEasyGpu")
    ];

    [Theory]
    [MemberData(nameof(ParityCases))]
    [Trait("Category", "Gpu")]
    public async Task ExistingLuisaParityCaseRunsThroughMetal(
        string fullyQualifiedName,
        string? expectedCompilerFailure)
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        var assemblyPath = typeof(LuisaBackendMetalTests).Assembly.Location;
        var startInfo = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        startInfo.ArgumentList.Add("test");
        startInfo.ArgumentList.Add(assemblyPath);
        startInfo.ArgumentList.Add("--filter");
        startInfo.ArgumentList.Add($"FullyQualifiedName={fullyQualifiedName}");
        startInfo.ArgumentList.Add("--logger");
        startInfo.ArgumentList.Add("console;verbosity=normal");
        startInfo.Environment["FEATHER_LUISA_BACKEND"] = "metal";
        startInfo.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1";

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start the isolated Metal parity test process.");
        var standardOutput = process.StandardOutput.ReadToEndAsync();
        var standardError = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        var output = string.Concat(await standardOutput, await standardError);

        if (expectedCompilerFailure is null)
        {
            Assert.True(process.ExitCode == 0,
                $"Metal parity case {fullyQualifiedName} failed with exit {process.ExitCode}:{Environment.NewLine}{output}");
            return;
        }

        Assert.NotEqual(0, process.ExitCode);
        Assert.Contains("Test Run Aborted", output, StringComparison.Ordinal);
        Assert.Contains(expectedCompilerFailure, output, StringComparison.Ordinal);
        Assert.Contains("metal_compiler.cpp:402", output, StringComparison.Ordinal);
    }

    private static object?[] Case(string name, string? expectedCompilerFailure = null)
        => [Prefix + name, expectedCompilerFailure];
}
