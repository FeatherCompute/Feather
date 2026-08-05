using System.Diagnostics;

namespace Feather.Integration.Tests;

public class LuisaBackendMetalTests
{
    private const string Prefix = "Feather.Integration.Tests.";
    private const string TextureFailure = "no member named 'sample'";
    private const string SwizzleFailure = "non-const reference cannot bind to vector element";

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
        // LC emits texture_sample* calls through metal_codegen_ast.cpp:1161-1173, while
        // metal_device_lib.metal:262-277 instantiates .sample() on access::read textures.
        Case("LuisaBackendResourceDispatchTests.TextureSamplingAndMixedResourceOrderMatchEasyGpu", TextureFailure),
        Case("LuisaBackendResourceDispatchTests.TextureSampleGradExecutesThroughLuisaXir", TextureFailure),
        Case("LuisaBackendResourceDispatchTests.NestedScalarCallablesMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.ShaderLibraryBufferCallablesMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.ShaderLibraryTextureAndSamplerCallablesMatchEasyGpu", TextureFailure),
        Case("LuisaBackendResourceDispatchTests.MutableGpuStructCallablesAndWritebackMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.StructArraysAndNestedWritebackMatchEasyGpu"),
        Case("LuisaBackendResourceDispatchTests.NonDivisibleLogicalBoundsMatchEasyGpu"),
        Case("LuisaBackendTests.VectorAddExecutesThroughLuisaXirVulkan"),
        Case("LuisaBackendTypeFeatureTests.ScalarsComparisonsLogicAndBitOperationsMatchEasyGpu"),
        Case("LuisaBackendTypeFeatureTests.UnsignedConstantsBitOperationsComparisonsAndConversionsMatchEasyGpu"),
        // LC emits swizzle temporaries in metal_codegen_ast.cpp:791-812, then passes them
        // to the non-const vector_element_ref overload in metal_device_lib.metal:191-194.
        Case("LuisaBackendTypeFeatureTests.VectorConstructionAndSwizzlesMatchEasyGpu", SwizzleFailure),
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
