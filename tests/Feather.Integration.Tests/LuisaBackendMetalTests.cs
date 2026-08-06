using System.Diagnostics;

namespace Feather.Integration.Tests;

public class LuisaBackendMetalTests
{
    private const string Prefix = "Feather.Integration.Tests.";

    public static IEnumerable<object?[]> ParityCases =>
    [
        Case("LuisaBackendAdTests.ReverseModeGradientsStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendAdTests.VectorParameterGradientsStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendAdTests.CallableGradientsExecuteThroughLuisa"),
        Case("LuisaBackendAdTests.ConditionalIntrinsicGradientsStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendControlMemoryTests.StructuredControlFlowAndMutableLocalsStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendControlMemoryTests.SharedMemoryLocalIdsAndBarrierStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.FullIntegerAtomicMatrixMatchesDefaultLuisa"),
        Case("LuisaBackendResourceDispatchTests.TwoAndThreeDimensionalDispatchIdsStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.Texture2DAndTexture3DLoadStoreStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.TextureSamplingAndMixedResourceOrderStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.TextureSampleGradExecutesThroughLuisaXir"),
        Case("LuisaBackendResourceDispatchTests.NestedScalarCallablesStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.ShaderLibraryBufferCallablesStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.ShaderLibraryTextureAndSamplerCallablesStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.MutableGpuStructCallablesAndWritebackStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.StructArraysAndNestedWritebackStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendResourceDispatchTests.NonDivisibleLogicalBoundsStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendTests.VectorAddExecutesThroughLuisaXirVulkan"),
        Case("LuisaBackendTypeFeatureTests.ScalarsComparisonsLogicAndBitOperationsStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendTypeFeatureTests.UnsignedConstantsBitOperationsComparisonsAndConversionsStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendTypeFeatureTests.VectorConstructionAndSwizzlesStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendTypeFeatureTests.MatrixConstructionAndLinearAlgebraStaticAndExplicitLuisaAgree"),
        Case("LuisaBackendTypeFeatureTests.StructAggregateLoadsAndNestedFieldExtractionStaticAndExplicitLuisaAgree")
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
