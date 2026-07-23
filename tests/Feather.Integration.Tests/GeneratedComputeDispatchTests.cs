using System.Buffers.Binary;
using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather.Integration.Tests;

/// <summary>
/// Typed-only tests in this file prove the canonical Roslyn typed-IR path.
/// Tests marked NativeReferenceFallback exercise compatibility/reference execution only.
/// </summary>
public class GeneratedComputeDispatchTests
{
    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchCopiesBufferThroughNativeFallback()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripSection7ForNativeReferenceFallback;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);
            var kernel = new CopyKernel(input.AsReadOnly(), output.AsReadWrite());

            GPU.Dispatch(kernel, 4);

            Assert.Equal([1, 2, 3, 4], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionReturnsEasyGpuBuilderSourceForCopyKernel()
    {
        var glsl = ShaderInspection.GetGLSL<CopyKernel>();

        Assert.Contains("#version", glsl, StringComparison.Ordinal);
        Assert.Contains("layout(local_size_x = 1", glsl, StringComparison.Ordinal);
        Assert.Contains("buffer fe_0_t", glsl, StringComparison.Ordinal);
        Assert.Contains("buffer fe_1_t", glsl, StringComparison.Ordinal);
        Assert.Contains("fe_1", glsl, StringComparison.Ordinal);
        Assert.Contains("fe_0", glsl, StringComparison.Ordinal);
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ShaderInspectionSanitizesReservedGlslLocalIdentifiers()
    {
        var glsl = ShaderInspection.GetGLSL<ReservedLocalIdentifierKernel>();

        Assert.Contains("float fe_output", glsl, StringComparison.Ordinal);
        Assert.Contains("float fe_texture", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("float output", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("float texture", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ShaderInspectionOptimizedPathUsesEasyGpuBackend()
    {
        try
        {
            var glsl = ShaderInspection.GetOptimizedGLSL<CopyKernel>();

            Assert.Contains("#version", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        catch (FeatherNativeException ex) when (
            ex.Result is FeResult.ErrorBackendUnavailable or FeResult.ErrorUnsupported or FeResult.ErrorShaderCompileFailed)
        {
            Assert.Contains("EasyGPU", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ShaderInspectionLowersUniformExpressionThroughEasyGpuPushConstants()
    {
        var glsl = ShaderInspection.GetGLSL<UniformExpressionKernel>();

        Assert.Contains("layout(push_constant) uniform EasyGPUUniformBlock", glsl, StringComparison.Ordinal);
        Assert.Contains("float u0;", glsl, StringComparison.Ordinal);
        Assert.Contains("fe_0", glsl, StringComparison.Ordinal);
        Assert.Contains(")*(u0)", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ShaderInspectionBuildsCopyKernelFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<CopyKernel>();

            Assert.Contains("#version", glsl, StringComparison.Ordinal);
            Assert.Contains("fe_1", glsl, StringComparison.Ordinal);
            Assert.Contains("fe_0", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesEntryAttributedComputeKernel()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);

        GPU.Dispatch(new EntryAttributedComputeKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

        Assert.Equal([4, 5, 6, 7], output.ToArray());
    }

    [Fact]
    public void DispatchExecutesRecommendedEntryKernelWithVarLocalsThroughTypedEasyGpu()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);

        var path = DispatchAndGetPath(new EntryVarLocalsKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<float>(1.5f)), 4);

        Assert.Equal([1.5f, 4.5f, 7.5f, 10.5f], output.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, path);
    }

    [Fact]
    public void DispatchSurvivesVulkanDescriptorPoolRollover()
    {
        const int dispatchCount = 1030;
        var inputs = new GpuBuffer<float>[dispatchCount];
        var outputs = new GpuBuffer<float>[dispatchCount];

        try
        {
            using var gpuKernel = GpuKernel.Create<UniformExpressionKernel>(GPU.Context);

            for (var i = 0; i < dispatchCount; i++)
            {
                inputs[i] = GPU.CreateBuffer<float>([(float)i]);
                outputs[i] = GPU.CreateBuffer<float>(1);
                var kernel = new UniformExpressionKernel(inputs[i].AsReadOnly(), outputs[i].AsReadWrite(), new Uniform<float>(2));

                GpuKernel.Dispatch(GPU.Context, gpuKernel, kernel, new GpuDispatchSize(1, 1, 1), wait: true);
            }

            Assert.Equal(DispatchPath.TypedEasyGpu, gpuKernel.LastDispatchPath);
            Assert.Equal((dispatchCount - 1) * 2f, outputs[^1].ToArray()[0]);
        }
        finally
        {
            foreach (var output in outputs)
            {
                output?.Dispose();
            }

            foreach (var input in inputs)
            {
                input?.Dispose();
            }
        }
    }

    [Fact]
    public void ShaderInspectionBuildsLiteralArithmeticFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<MultiplyFloatByLiteralKernel>();

            Assert.Contains("#version", glsl, StringComparison.Ordinal);
            Assert.Contains(")*(2", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionFusesFloatingPointMultiplyAddFromTypedIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var scalar = ShaderInspection.GetGLSL<NestedExpressionKernel>();
            var vector = ShaderInspection.GetGLSL<VectorFusedMultiplyAddKernel>();

            Assert.Contains("fma(", scalar, StringComparison.Ordinal);
            Assert.Contains("fma(", vector, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionDoesNotFuseIntegerMultiplyAddFromTypedIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<IntegerMultiplyAddKernel>();

            Assert.DoesNotContain("fma(", glsl, StringComparison.Ordinal);
            Assert.Contains(")*(", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsUnsignedIntegerArithmeticFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<UnsignedIntegerKernel>();

            Assert.Contains("uint", glsl, StringComparison.Ordinal);
            Assert.Contains("fe_1", glsl, StringComparison.Ordinal);
            Assert.Contains("fe_0", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesUnsignedIntegerArithmeticFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<uint>([1u, 2u, 3u, 4u]);
            using var output = GPU.CreateBuffer<uint>(4);

            GPU.Dispatch(new UnsignedIntegerKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal([5u, 7u, 9u, 11u], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesUnsignedUniformFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<uint>([1u, 2u, 3u, 4u]);
            using var output = GPU.CreateBuffer<uint>(4);

            GPU.Dispatch(new UnsignedUniformKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<uint>(7u)), 4);

            Assert.Equal([8u, 9u, 10u, 11u], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsLocalThreadAliasFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<CopyKernel>();

            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("ASSIGN1", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsPushConstantExpressionFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<UniformExpressionKernel>();

            Assert.Contains("layout(push_constant) uniform EasyGPUUniformBlock", glsl, StringComparison.Ordinal);
            Assert.Contains(")*(u0)", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchCopiesBufferFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new CopyKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([1, 2, 3, 4], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionEmitsLogicalBoundsGuardForBoundsCheckedKernels()
    {
        var glsl = ShaderInspection.GetGLSL<BoundsCheckedWriteKernel>();

        Assert.Contains("if ", glsl, StringComparison.Ordinal);
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
        Assert.Contains(">=", glsl, StringComparison.Ordinal);
        Assert.Contains("return;", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ShaderInspectionDoesNotEmitLogicalBoundsGuardWhenDisabled()
    {
        var glsl = ShaderInspection.GetGLSL<UncheckedBoundsWriteKernel>();

        Assert.DoesNotContain("return;", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain(">=", glsl, StringComparison.Ordinal);
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void DispatchBoundsCheckProtectsNonDivisibleOneDimensionalDispatch()
    {
        using var output = GPU.CreateBuffer<int>(Enumerable.Repeat(-1, 16).ToArray());

        var path = DispatchAndGetPath(new BoundsCheckedWriteKernel(output.AsReadWrite()), 5);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        Assert.Equal([0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], output.ToArray());
    }

    [Fact]
    public void DispatchBoundsCheckProtectsNonDivisibleTwoDimensionalDispatch()
    {
        using var output = GPU.CreateBuffer<int>(Enumerable.Repeat(-1, 16).ToArray());

        var path = DispatchAndGetPath(new BoundsChecked2DWriteKernel(output.AsReadWrite()), new GpuDispatchSize(3, 2, 1));

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        Assert.Equal([0, 1, 2, -1, 10, 11, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1], output.ToArray());
    }

    [Fact]
    public void DispatchBoundsCheckProtectsNonDivisibleThreeDimensionalDispatch()
    {
        using var output = GPU.CreateBuffer<int>(Enumerable.Repeat(-1, 32).ToArray());

        var path = DispatchAndGetPath(new BoundsChecked3DWriteKernel(output.AsReadWrite()), new GpuDispatchSize(3, 2, 2));

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        Assert.Equal([
            0, 1, 2, -1,
            10, 11, 12, -1,
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            100, 101, 102, -1,
            110, 111, 112, -1,
            -1, -1, -1, -1,
            -1, -1, -1, -1
        ], output.ToArray());
    }

    [Fact]
    public void DispatchMultipliesLiteralFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new MultiplyFloatByLiteralKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([2, 4, 6, 8], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsIfElseFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<IfThresholdKernel>();

            Assert.Contains("#version", glsl, StringComparison.Ordinal);
            Assert.Contains("if ", glsl, StringComparison.Ordinal);
            Assert.Contains("else", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("if (true)", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesIfElseFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([-2, -1, 0, 1, 2, 3]);
            using var output = GPU.CreateBuffer<float>(6);

            GPU.Dispatch(new IfThresholdKernel(input.AsReadOnly(), output.AsReadWrite()), 6);

            Assert.Equal([0, 0, 0, 1, 2, 3], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsDynamicForLoopFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<DynamicForSumKernel>();

            Assert.Contains("#version", glsl, StringComparison.Ordinal);
            Assert.Contains("for ", glsl, StringComparison.Ordinal);
            Assert.Contains("fe_0", glsl, StringComparison.Ordinal);
            Assert.Contains("fe_1", glsl, StringComparison.Ordinal);
            Assert.Contains("fe_2", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.Contains("[i]", glsl, StringComparison.Ordinal);
            Assert.Contains("i)+(j", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("for (;", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesDynamicForLoopFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var counts = GPU.CreateBuffer<int>([1, 2, 3, 4]);
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4, 5, 6, 7]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new DynamicForSumKernel(counts.AsReadOnly(), input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([1, 5, 12, 22], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesWhileLoopFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(3);

            GPU.Dispatch(new WhileAccumulateKernel(input.AsReadOnly(), output.AsReadWrite()), 3);

            Assert.Equal([10, 10, 10], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesDoWhileFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([2, 3, 5, 7, 11]);
            using var output = GPU.CreateBuffer<float>(3);

            GPU.Dispatch(new DoWhileAccumulateKernel(input.AsReadOnly(), output.AsReadWrite()), 3);

            Assert.Equal([10, 15, 23], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesBreakAndContinueFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 10, 100, 1000, 10000]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new BreakContinueLoopKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([101, 101, 101, 101], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesConditionlessForWithBreakFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([3, 4, 5, 6]);
            using var output = GPU.CreateBuffer<float>(4);

            GPU.Dispatch(new ConditionlessForBreakKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal([6, 8, 10, 12], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesLogicalPredicateFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([-2, -1, 0, 1, 2, 3]);
            using var output = GPU.CreateBuffer<float>(6);

            GPU.Dispatch(new LogicalPredicateKernel(input.AsReadOnly(), output.AsReadWrite()), 6);

            Assert.Equal([-2, 0, 0, 0, 2, 3], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesUnaryAndModuloFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<int>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<int>(4);

            GPU.Dispatch(new UnaryModuloKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal([-1, -3, -3, -5], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesBitwiseAndShiftFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<int>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<int>(4);

            GPU.Dispatch(new BitwiseShiftKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal([2, 4, 6, 8], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesVectorConstructorFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateBuffer<float3>(3);

            GPU.Dispatch(new VectorConstructorKernel(output.AsReadWrite()), 3);

            Assert.Equal([
                new float3(0, 1, 2),
                new float3(1, 2, 3),
                new float3(2, 3, 4)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesScalarIntrinsicFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 4, 9, 16]);
            using var bias = GPU.CreateBuffer<float>([10, 20, 30, 40]);
            using var output = GPU.CreateBuffer<float>(4);

            GPU.Dispatch(new IntrinsicExpressionKernel(input.AsReadOnly(), bias.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal([11, 22, 33, 44], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesVectorIntrinsicsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var left = GPU.CreateBuffer<float3>([new float3(1, 2, 3), new float3(0, 1, 0)]);
            using var right = GPU.CreateBuffer<float3>([new float3(7, 8, 9), new float3(0, 0, 1)]);
            using var dot = GPU.CreateBuffer<float>(2);
            using var cross = GPU.CreateBuffer<float3>(2);

            GPU.Dispatch(new DotExpressionKernel(left.AsReadOnly(), right.AsReadOnly(), dot.AsReadWrite()), 2);
            GPU.Dispatch(new CrossExpressionKernel(left.AsReadOnly(), right.AsReadOnly(), cross.AsReadWrite()), 2);

            Assert.Equal([50, 0], dot.ToArray());
            Assert.Equal([new float3(-6, 12, -6), new float3(1, 0, 0)], cross.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesAdditionalScalarIntrinsicsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([4.25f, 9.75f, 16.5f, 25.0f]);
            using var output = GPU.CreateBuffer<float>(4);

            GPU.Dispatch(new AdditionalScalarIntrinsicKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            AssertNear([
                (1.0f / MathF.Sqrt(4.25f)) + 0.25f,
                (1.0f / MathF.Sqrt(9.75f)) + 0.75f,
                (1.0f / MathF.Sqrt(16.5f)) + 0.5f,
                (1.0f / MathF.Sqrt(25.0f)) + 1.0f
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesLengthAndNormalizeFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float3>([
                new float3(3, 0, 0),
                new float3(0, 4, 0),
                new float3(0, 0, 5)
            ]);
            using var normalized = GPU.CreateBuffer<float3>(3);
            using var lengths = GPU.CreateBuffer<float>(3);

            GPU.Dispatch(new NormalizeIntrinsicKernel(input.AsReadOnly(), normalized.AsReadWrite()), 3);
            GPU.Dispatch(new LengthIntrinsicKernel(input.AsReadOnly(), lengths.AsReadWrite()), 3);

            Assert.Equal([new float3(1, 0, 0), new float3(0, 1, 0), new float3(0, 0, 1)], normalized.ToArray());
            Assert.Equal([3, 4, 5], lengths.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesVectorMathIntrinsicOverloadsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var left = GPU.CreateBuffer<float3>([
                new float3(-2.0f, 0.25f, 4.0f),
                new float3(0.2f, -4.0f, 0.8f)
            ]);
            using var right = GPU.CreateBuffer<float3>([
                new float3(0.0f, 2.0f, 0.5f),
                new float3(1.0f, 0.0f, 0.3f)
            ]);
            using var output = GPU.CreateBuffer<float3>(2);

            GPU.Dispatch(new VectorMathIntrinsicOverloadKernel(left.AsReadOnly(), right.AsReadOnly(), output.AsReadWrite()), 2);

            Assert.Equal([
                new float3(0.0f, 1.125f, 0.5f),
                new float3(0.6f, 0.0f, 0.3f)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsVectorSwizzlesFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<SwizzleReadKernel>();

            Assert.Contains(".xy", glsl, StringComparison.Ordinal);
            Assert.Contains(".z", glsl, StringComparison.Ordinal);
            Assert.Contains(".w", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesVectorSwizzlesFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float4>([
                new float4(1, 2, 3, 4),
                new float4(5, 6, 7, 8)
            ]);
            using var output = GPU.CreateBuffer<float3>(2);

            GPU.Dispatch(new SwizzleReadKernel(input.AsReadOnly(), output.AsReadWrite()), 2);

            Assert.Equal([new float3(1, 2, 7), new float3(5, 6, 15)], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesDirectResourceColorSwizzleFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float4>([
                new float4(2, 4, 6, 8),
                new float4(1, 3, 5, 7)
            ]);
            using var output = GPU.CreateBuffer<float3>(2);

            GPU.Dispatch(new DirectColorSwizzleKernel(input.AsReadOnly(), output.AsReadWrite()), 2);

            Assert.Equal([new float3(2, 4, 6), new float3(1, 3, 5)], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsExpandedVectorSwizzlesFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<ExpandedSwizzleKernel>();

            Assert.Contains(".yx", glsl, StringComparison.Ordinal);
            Assert.Contains(".zxy", glsl, StringComparison.Ordinal);
            Assert.Contains(".wzyx", glsl, StringComparison.Ordinal);
            Assert.Contains(".zyxw", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesExpandedVectorSwizzlesFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float4>([
                new float4(1, 2, 3, 4),
                new float4(5, 6, 7, 8)
            ]);
            using var output = GPU.CreateBuffer<float4>(2);

            GPU.Dispatch(new ExpandedSwizzleKernel(input.AsReadOnly(), output.AsReadWrite()), 2);

            Assert.Equal([
                new float4(5, 2, 7, 5),
                new float4(13, 10, 15, 13)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsTextureCoordinateSwizzlesFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<TextureCoordinateSwizzleKernel>();

            Assert.Contains(".yx", glsl, StringComparison.Ordinal);
            Assert.Contains(".zxy", glsl, StringComparison.Ordinal);
            Assert.Contains(".wzyx", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesTextureCoordinateSwizzlesFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float4>([
                new float4(1, 2, 3, 4),
                new float4(5, 6, 7, 8)
            ]);
            using var output = GPU.CreateBuffer<float4>(2);

            GPU.Dispatch(new TextureCoordinateSwizzleKernel(input.AsReadOnly(), output.AsReadWrite()), 2);

            Assert.Equal([
                new float4(2, 1, 3, 1),
                new float4(6, 5, 7, 5)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsSwizzleWriteLValueFromTypedIr()
    {
        var glsl = ShaderInspection.GetGLSL<RawSwizzleWriteKernel>();

        Assert.Contains(".xy", glsl, StringComparison.Ordinal);
        Assert.Contains("vec2", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void DispatchExecutesSwizzleWriteBackFromTypedIr()
    {
        using var output = GPU.CreateBuffer<float4>([
            new float4(2, 4, 6, 8)
        ]);

        var path = DispatchAndGetPath(new RawSwizzleWriteKernel(output.AsReadWrite()), 1);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        Assert.Equal([new float4(1, 1, 6, 8)], output.ToArray());
    }

    [Fact]
    public void ShaderInspectionRejectsDuplicateComponentSwizzleWriteFromTypedIr()
    {
        var exception = Assert.Throws<FeatherNativeException>(() => ShaderInspection.GetGLSL<RawDuplicateSwizzleWriteKernel>());

        Assert.Equal(FeResult.ErrorUnsupported, exception.Result);
        Assert.Contains("swizzle l-value", exception.Message, StringComparison.Ordinal);
        Assert.Contains("same component more than once", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DispatchExecutesIntegerVectorConstructorFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateBuffer<int3>(4);

            GPU.Dispatch(new IntegerVectorConstructorKernel(output.AsReadWrite()), 4);

            Assert.Equal([
                new int3(0, 10, 0),
                new int3(1, 11, 2),
                new int3(2, 12, 4),
                new int3(3, 13, 6)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsBoolVectorConstructorsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<BoolVectorConstructorKernel>();

            Assert.Contains("bvec2", glsl, StringComparison.Ordinal);
            Assert.Contains("bvec3", glsl, StringComparison.Ordinal);
            Assert.Contains("bvec4", glsl, StringComparison.Ordinal);
            Assert.Contains(".x", glsl, StringComparison.Ordinal);
            Assert.Contains(".z", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesBoolVectorConstructorsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateBuffer<int>(4);

            GPU.Dispatch(new BoolVectorConstructorKernel(output.AsReadWrite()), 4);

            Assert.Equal([7, 12, 14, 10], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsNumericCastsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var scalar = ShaderInspection.GetGLSL<NumericCastKernel>();
            var vector = ShaderInspection.GetGLSL<VectorCastKernel>();

            Assert.Contains("float(", scalar, StringComparison.Ordinal);
            Assert.Contains("int(", scalar, StringComparison.Ordinal);
            Assert.Contains("vec3(", vector, StringComparison.Ordinal);
            Assert.Contains("ivec3(", vector, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", scalar, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", vector, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesNumericCastsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var ints = GPU.CreateBuffer<int>([1, -2, 3, 4]);
            using var floats = GPU.CreateBuffer<float>([1.75f, -2.25f, 3.9f, 4.1f]);
            using var scalarOutput = GPU.CreateBuffer<float>(4);
            using var vectorOutput = GPU.CreateBuffer<int3>(4);

            GPU.Dispatch(new NumericCastKernel(ints.AsReadOnly(), floats.AsReadOnly(), scalarOutput.AsReadWrite()), 4);
            GPU.Dispatch(new VectorCastKernel(ints.AsReadOnly(), vectorOutput.AsReadWrite()), 4);

            Assert.Equal([2, -4, 6, 8], scalarOutput.ToArray());
            Assert.Equal([
                new int3(1, 2, 3),
                new int3(-2, -1, 0),
                new int3(3, 4, 5),
                new int3(4, 5, 6)
            ], vectorOutput.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsMatrixConstructorsAndOperatorsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var mat2 = ShaderInspection.GetGLSL<Matrix2VectorMultiplyKernel>();
            var mat3 = ShaderInspection.GetGLSL<Matrix3MultiplyKernel>();
            var mat4 = ShaderInspection.GetGLSL<Matrix4VectorMultiplyKernel>();

            Assert.Contains("mat2", mat2, StringComparison.Ordinal);
            Assert.Contains("vec2", mat2, StringComparison.Ordinal);
            Assert.Contains("*", mat2, StringComparison.Ordinal);
            Assert.Contains("mat3", mat3, StringComparison.Ordinal);
            Assert.Contains("mat4", mat4, StringComparison.Ordinal);
            Assert.Contains("vec4", mat4, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", mat2, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", mat3, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", mat4, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsMatrixColumnAccessFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<MatrixColumnAccessKernel>();

            Assert.Contains("mat4", glsl, StringComparison.Ordinal);
            Assert.Contains("[2]", glsl, StringComparison.Ordinal);
            Assert.Contains(".xyz", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesMatrixVectorMultiplyFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateBuffer<float2>(3);

            GPU.Dispatch(new Matrix2VectorMultiplyKernel(output.AsReadWrite()), 3);

            Assert.Equal([
                new float2(3, 4),
                new float2(7, 10),
                new float2(11, 16)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesMatrixMultiplyAndColumnAccessFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var matrix3Output = GPU.CreateBuffer<float3>(3);
            using var columnOutput = GPU.CreateBuffer<float3>(3);
            using var matrix4Output = GPU.CreateBuffer<float4>(3);

            GPU.Dispatch(new Matrix3MultiplyKernel(matrix3Output.AsReadWrite()), 3);
            GPU.Dispatch(new MatrixColumnAccessKernel(columnOutput.AsReadWrite()), 3);
            GPU.Dispatch(new Matrix4VectorMultiplyKernel(matrix4Output.AsReadWrite()), 3);

            Assert.Equal([
                new float3(0, 2, 0),
                new float3(0, 3, 0),
                new float3(0, 4, 0)
            ], matrix3Output.ToArray());
            Assert.Equal([
                new float3(0, 1, 2),
                new float3(1, 2, 3),
                new float3(2, 3, 4)
            ], columnOutput.ToArray());
            Assert.Equal([
                new float4(1, 3, 5, 1),
                new float4(2, 4, 6, 1),
                new float4(3, 5, 7, 1)
            ], matrix4Output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsMatrixMathIntrinsicsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<MatrixMathIntrinsicKernel>();

            Assert.Contains("transpose(", glsl, StringComparison.Ordinal);
            Assert.Contains("determinant(", glsl, StringComparison.Ordinal);
            Assert.Contains("inverse(", glsl, StringComparison.Ordinal);
            Assert.Contains("matrixCompMult(", glsl, StringComparison.Ordinal);
            Assert.Contains("*", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsMatrixPushConstantsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<Matrix3UniformMultiplyKernel>();

            Assert.Contains("layout(push_constant) uniform EasyGPUUniformBlock", glsl, StringComparison.Ordinal);
            Assert.Contains("mat3 u0;", glsl, StringComparison.Ordinal);
            Assert.Contains("mat3 u1;", glsl, StringComparison.Ordinal);
            Assert.Contains("float u2;", glsl, StringComparison.Ordinal);
            Assert.Contains("mat3", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesMatrixPushConstantsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateBuffer<float3>(4);
            var transform = new float3x3(
                new float3(1, 2, 3),
                new float3(4, 5, 6),
                new float3(7, 8, 9));
            var offsetTransform = new float3x3(
                new float3(0.5f, 1.0f, 1.5f),
                new float3(-1.0f, -2.0f, -3.0f),
                new float3(2.0f, 4.0f, 6.0f));

            var path = DispatchAndGetPath(
                new Matrix3UniformMultiplyKernel(
                    output.AsReadWrite(),
                    new Uniform<float3x3>(transform),
                    new Uniform<float3x3>(offsetTransform),
                    new Uniform<float>(2.0f)),
                4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([
                new float3(12, 15, 18),
                new float3(17, 22, 27),
                new float3(22, 29, 36),
                new float3(27, 36, 45)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsSquareMatrixPushConstantsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<SquareMatrixUniformKernel>();

            Assert.Contains("layout(push_constant) uniform EasyGPUUniformBlock", glsl, StringComparison.Ordinal);
            Assert.Contains("mat2 u0;", glsl, StringComparison.Ordinal);
            Assert.Contains("mat3 u1;", glsl, StringComparison.Ordinal);
            Assert.Contains("mat4 u2;", glsl, StringComparison.Ordinal);
            Assert.Contains("float u3;", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesSquareMatrixPushConstantsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateBuffer<float4>(4);

            var matrix2 = new float2x2(
                new float2(1, 2),
                new float2(3, 4));
            var matrix3 = new float3x3(
                new float3(1, 0, 0),
                new float3(0, 2, 0),
                new float3(5, 6, 7));
            var matrix4 = new float4x4(
                new float4(1, 0, 0, 0),
                new float4(0, 1, 0, 0),
                new float4(0, 0, 2, 0),
                new float4(10, 20, 30, 1));

            var path = DispatchAndGetPath(
                new SquareMatrixUniformKernel(
                    output.AsReadWrite(),
                    new Uniform<float2x2>(matrix2),
                    new Uniform<float3x3>(matrix3),
                    new Uniform<float4x4>(matrix4),
                    new Uniform<float>(3.0f)),
                4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([
                new float4(13, 18, 36, 3),
                new float4(14, 22, 36, 3),
                new float4(15, 26, 36, 3),
                new float4(16, 30, 36, 3)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesMatrixMathIntrinsicsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateBuffer<float4>(3);

            GPU.Dispatch(new MatrixMathIntrinsicKernel(output.AsReadWrite()), 3);

            Assert.Equal([
                new float4(8, 16, 32, 1),
                new float4(16, 32, 64, 2),
                new float4(24, 48, 96, 3)
            ], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesThreadIdsXyFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<ThreadIdsXyLinearIndexKernel>();
            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.y", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("gl_GlobalInvocationID.z", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

            using var output = GPU.CreateBuffer<int>(6);

            GPU.Dispatch(new ThreadIdsXyLinearIndexKernel(output.AsReadWrite()), new int2(3, 2));

            Assert.Equal([0, 1, 2, 10, 11, 12], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesThreadIdsXyzFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<ThreadIdsXyzLinearIndexKernel>();
            Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.y", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_GlobalInvocationID.z", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

            using var output = GPU.CreateBuffer<int>(8);

            GPU.Dispatch(new ThreadIdsXyzLinearIndexKernel(output.AsReadWrite()), new int3(2, 2, 2));

            Assert.Equal([0, 1, 10, 11, 100, 101, 110, 111], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsLocalIdsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<LocalIdsKernel>();

            Assert.Contains("gl_LocalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsTypedSharedFloatMemoryAndBarriersWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<SharedFloatCopyKernel>();
            var writeIndex = glsl.IndexOf("shared_values", StringComparison.Ordinal);
            var barrierIndex = glsl.IndexOf("barrier();", StringComparison.Ordinal);
            var readIndex = glsl.LastIndexOf("shared_values", StringComparison.Ordinal);

            Assert.Contains("shared float shared_values[4];", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_LocalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.True(writeIndex >= 0, glsl);
            Assert.True(barrierIndex > writeIndex, glsl);
            Assert.True(readIndex > barrierIndex, glsl);
            Assert.DoesNotContain("shared_data", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsTypedSharedIntMemoryWithDynamicLocalIndexWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<SharedIntDynamicIndexKernel>();

            Assert.Contains("shared int shared_ints[8];", glsl, StringComparison.Ordinal);
            Assert.Contains("shared_ints", glsl, StringComparison.Ordinal);
            Assert.Contains("slot", glsl, StringComparison.Ordinal);
            Assert.Contains("gl_LocalInvocationID.x", glsl, StringComparison.Ordinal);
            Assert.Contains("barrier();", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("shared_data", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsMultipleTypedSharedMemoriesWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<MultipleSharedMemoryKernel>();

            Assert.Contains("shared float shared_left[4];", glsl, StringComparison.Ordinal);
            Assert.Contains("shared int shared_right[4];", glsl, StringComparison.Ordinal);
            Assert.Contains("shared_left", glsl, StringComparison.Ordinal);
            Assert.Contains("shared_right", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("shared_data", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsVectorSharedMemoryWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<SharedVectorMemoryKernel>();

            Assert.Contains("shared vec2 shared_vectors[2];", glsl, StringComparison.Ordinal);
            Assert.Contains("vec2", glsl, StringComparison.Ordinal);
            Assert.Contains("shared_vectors", glsl, StringComparison.Ordinal);
            Assert.Contains("barrier();", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("shared_data", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionPreservesTypedBarrierKindsWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<TypedBarrierKindsKernel>();
            var firstWorkgroup = glsl.IndexOf("barrier();", StringComparison.Ordinal);
            var memory = glsl.IndexOf("memoryBarrier();", StringComparison.Ordinal);
            var secondMemory = glsl.IndexOf("memoryBarrier();", memory + 1, StringComparison.Ordinal);
            var fullBarrier = glsl.IndexOf("barrier();", secondMemory + 1, StringComparison.Ordinal);

            Assert.True(firstWorkgroup >= 0, glsl);
            Assert.True(memory > firstWorkgroup, glsl);
            Assert.True(secondMemory > memory, glsl);
            Assert.True(fullBarrier > secondMemory, glsl);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsCallableFunctionFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<TypedCallableKernel>();
            const string callableName = "global__Feather_Integration_Tests_TypedCallableKernel_Twice_";
            const string callableDefinitionPrefix = "float global__Feather_Integration_Tests_TypedCallableKernel_Twice_";
            var mainStart = glsl.IndexOf("void main()", StringComparison.Ordinal);
            var callableCallStart = mainStart < 0
                ? -1
                : glsl.IndexOf(callableName, mainStart, StringComparison.Ordinal);
            var callableBodyStart = mainStart < 0
                ? -1
                : glsl.IndexOf(callableDefinitionPrefix, mainStart, StringComparison.Ordinal);

            Assert.Contains(callableDefinitionPrefix, glsl, StringComparison.Ordinal);
            Assert.True(mainStart >= 0, "GLSL must contain an entry-point main function.");
            Assert.True(callableBodyStart > mainStart, "Callable definition should be emitted after main with a forward declaration.");
            Assert.True(callableCallStart > mainStart && callableCallStart < callableBodyStart, "Main should call the forwarded callable before its definition.");
            Assert.Contains("fe_0", glsl, StringComparison.Ordinal);
            Assert.Contains("fe_1", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesCallableFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new TypedCallableKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([2, 4, 6, 8], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesCallableWithControlFlowFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([-2, 2, 6, 12]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new TypedCallableControlFlowKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([-2, 6, 18, 18], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesCallableParameterReassignmentFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 2, -1, 4]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new TypedCallableParameterReassignmentKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([3, 9, 1, 13], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesOverloadedCallableByMangledIdentityFromTypedIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new TypedCallableOverloadKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([2, 4, 6, 8], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesSharedMemoryCopyFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);

            var path = DispatchAndGetPath(new SharedFloatCopyKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([4, 3, 2, 1], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesVectorSharedMemoryFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateBuffer<float2>(2);

            var path = DispatchAndGetPath(new SharedVectorMemoryKernel(output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([new float2(0, 1), new float2(1, 2)], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionRejectsTypedOnlyIrWhenSection7LoweringFails()
    {
        try
        {
            GpuKernel.IrTransformForTesting = CorruptTypedOnlyResourceName;

            var exception = Assert.Throws<FeatherNativeException>(() => ShaderInspection.GetGLSL<CopyKernel>());

            Assert.Equal(FeResult.ErrorUnsupported, exception.Result);
            Assert.Contains("unknown resource 'ghost'", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchRejectsTypedOnlyIrWhenSection7LoweringFails()
    {
        try
        {
            GpuKernel.IrTransformForTesting = CorruptTypedOnlyResourceName;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);

            using var gpuKernel = GpuKernel.Create<CopyKernel>(GPU.Context);
            var exception = Assert.Throws<FeatherNativeException>(
                () => GpuKernel.Dispatch(
                    GPU.Context,
                    gpuKernel,
                    new CopyKernel(input.AsReadOnly(), output.AsReadWrite()),
                    new GpuDispatchSize(4, 1, 1),
                    wait: true));

            Assert.Equal(FeResult.ErrorUnsupported, exception.Result);
            Assert.Contains("unknown resource 'ghost'", exception.Message, StringComparison.Ordinal);
            Assert.Equal(DispatchPath.Rejected, gpuKernel.LastDispatchPath);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchUsesStructuredAssignmentSectionWithoutAssignPayload()
    {
        try
        {
            GpuKernel.IrTransformForTesting = RemoveAssignmentCompatibilityOperands;
            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);
            var kernel = new CopyKernel(input.AsReadOnly(), output.AsReadWrite());

            GPU.Dispatch(kernel, 4);

            Assert.Equal([1, 2, 3, 4], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void Dispatch2DCopiesLinearBufferThroughNativeFallback()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripSection7ForNativeReferenceFallback;
            using var input = GPU.CreateBuffer<int>([1, 2, 3, 4, 5, 6]);
            using var output = GPU.CreateBuffer<int>(6);
            var kernel = new Copy2DKernel(input.AsReadOnly(), output.AsReadWrite());

            GPU.Dispatch(kernel, new int2(3, 2));

            Assert.Equal([1, 2, 3, 4, 5, 6], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void Dispatch3DCopiesLinearBufferThroughNativeFallback()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripSection7ForNativeReferenceFallback;
            using var input = GPU.CreateBuffer<int>([1, 2, 3, 4, 5, 6, 7, 8]);
            using var output = GPU.CreateBuffer<int>(8);
            var kernel = new Copy3DKernel(input.AsReadOnly(), output.AsReadWrite());

            GPU.Dispatch(kernel, new int3(2, 2, 2));

            Assert.Equal([1, 2, 3, 4, 5, 6, 7, 8], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchBindsUniformPushConstantsThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        var kernel = new CopyWithUniformKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<float>(3));

        GPU.Dispatch(kernel, 4);

        Assert.Equal([1, 2, 3, 4], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchReadsUniformValueInsideExpressionThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        var kernel = new UniformExpressionKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<float>(3));

        GPU.Dispatch(kernel, 4);

        Assert.Equal([3, 6, 9, 12], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchEvaluatesFloat2ExpressionThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float2>([new float2(1, 2), new float2(3, 4)]);
        using var bias = GPU.CreateBuffer<float2>([new float2(10, 20), new float2(30, 40)]);
        using var output = GPU.CreateBuffer<float2>(2);
        var kernel = new Float2ExpressionKernel(input.AsReadOnly(), bias.AsReadOnly(), output.AsReadWrite(), new Uniform<float>(2));

        GPU.Dispatch(kernel, 2);

        Assert.Equal([new float2(12, 24), new float2(36, 48)], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchReadsFloat2UniformInsideExpressionThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float2>([new float2(1, 2), new float2(3, 4)]);
        using var output = GPU.CreateBuffer<float2>(2);
        var kernel = new Float2UniformExpressionKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<float2>(new float2(10, 20)));

        GPU.Dispatch(kernel, 2);

        Assert.Equal([new float2(11, 22), new float2(13, 24)], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchEvaluatesFloat3ExpressionThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float3>([new float3(1, 2, 3), new float3(4, 5, 6)]);
        using var output = GPU.CreateBuffer<float3>(2);
        var kernel = new Float3ExpressionKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<float>(2));

        GPU.Dispatch(kernel, 2);

        Assert.Equal([new float3(2, 4, 6), new float3(8, 10, 12)], output.ToArray());
    }

    [Fact]
    public void Float3BufferRoundTripUsesEasyGpuStd430Stride()
    {
        using var buffer = GPU.CreateBuffer<float3>([new float3(1, 2, 3), new float3(4, 5, 6)]);

        Assert.Equal(2, buffer.Length);
        Assert.Equal(16, buffer.ElementStride);
        Assert.Equal(32, buffer.SizeInBytes);
        Assert.Equal([new float3(1, 2, 3), new float3(4, 5, 6)], buffer.ToArray());

        buffer.Upload(1, [new float3(7, 8, 9)]);

        Assert.Equal([new float3(1, 2, 3), new float3(7, 8, 9)], buffer.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchReadsFloat4UniformInsideExpressionThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float4>([new float4(1, 2, 3, 4), new float4(5, 6, 7, 8)]);
        using var output = GPU.CreateBuffer<float4>(2);
        var kernel = new Float4UniformExpressionKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<float4>(new float4(10, 20, 30, 40)));

        GPU.Dispatch(kernel, 2);

        Assert.Equal([new float4(11, 22, 33, 44), new float4(15, 26, 37, 48)], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchEvaluatesDotIntrinsicThroughNativeFallback()
    {
        using var left = GPU.CreateBuffer<float3>([new float3(1, 2, 3), new float3(4, 5, 6)]);
        using var right = GPU.CreateBuffer<float3>([new float3(7, 8, 9), new float3(1, 2, 3)]);
        using var output = GPU.CreateBuffer<float>(2);
        var kernel = new DotExpressionKernel(left.AsReadOnly(), right.AsReadOnly(), output.AsReadWrite());

        GPU.Dispatch(kernel, 2);

        Assert.Equal([50, 32], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchEvaluatesCrossIntrinsicThroughNativeFallback()
    {
        using var left = GPU.CreateBuffer<float3>([new float3(1, 0, 0), new float3(0, 1, 0)]);
        using var right = GPU.CreateBuffer<float3>([new float3(0, 1, 0), new float3(0, 0, 1)]);
        using var output = GPU.CreateBuffer<float3>(2);
        var kernel = new CrossExpressionKernel(left.AsReadOnly(), right.AsReadOnly(), output.AsReadWrite());

        GPU.Dispatch(kernel, 2);

        Assert.Equal([new float3(0, 0, 1), new float3(1, 0, 0)], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchMultipliesFloatBufferByLiteralThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        var kernel = new MultiplyFloatByLiteralKernel(input.AsReadOnly(), output.AsReadWrite());

        GPU.Dispatch(kernel, 4);

        Assert.Equal([2, 4, 6, 8], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchAddsIntegerBuffersThroughNativeFallback()
    {
        using var left = GPU.CreateBuffer<int>([1, 2, 3, 4]);
        using var right = GPU.CreateBuffer<int>([10, 20, 30, 40]);
        using var output = GPU.CreateBuffer<int>(4);
        var kernel = new AddIntegerBuffersKernel(left.AsReadOnly(), right.AsReadOnly(), output.AsReadWrite());

        GPU.Dispatch(kernel, 4);

        Assert.Equal([11, 22, 33, 44], output.ToArray());
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchEvaluatesNestedExpressionSectionThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var bias = GPU.CreateBuffer<float>([10, 20, 30, 40]);
        using var output = GPU.CreateBuffer<float>(4);
        var kernel = new NestedExpressionKernel(input.AsReadOnly(), bias.AsReadOnly(), output.AsReadWrite());

        GPU.Dispatch(kernel, 4);

        Assert.Equal([12, 24, 36, 48], output.ToArray());
    }

    [Fact]
    public void DispatchExecutesFusedMultiplyAddFromTypedIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var left = GPU.CreateBuffer<float3>([new float3(1, 2, 3), new float3(-2, 4, 0.5f)]);
            using var right = GPU.CreateBuffer<float3>([new float3(4, 5, 6), new float3(3, -1, 8)]);
            using var bias = GPU.CreateBuffer<float3>([new float3(0.5f, 1, 1.5f), new float3(1, 2, -3)]);
            using var output = GPU.CreateBuffer<float3>(2);

            GPU.Dispatch(
                new VectorFusedMultiplyAddKernel(
                    left.AsReadOnly(), right.AsReadOnly(), bias.AsReadOnly(), output.AsReadWrite()),
                2);

            Assert.Equal([new float3(4.5f, 11, 19.5f), new float3(-5, -2, 1)], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    [Trait("Coverage", "NativeReferenceFallback")]
    public void DispatchEvaluatesIntrinsicExpressionSectionThroughNativeFallback()
    {
        using var input = GPU.CreateBuffer<float>([1, 4, 9, 16]);
        using var bias = GPU.CreateBuffer<float>([10, 20, 30, 40]);
        using var output = GPU.CreateBuffer<float>(4);
        var kernel = new IntrinsicExpressionKernel(input.AsReadOnly(), bias.AsReadOnly(), output.AsReadWrite());

        GPU.Dispatch(kernel, 4);

        Assert.Equal([11, 22, 33, 44], output.ToArray());
    }

    [Fact]
    public void DispatchExecutesMultipleTypedAssignmentsInSourceOrder()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var bias = GPU.CreateBuffer<float>([10, 20, 30, 40]);
        using var expressionOutput = GPU.CreateBuffer<float>(4);
        using var copyOutput = GPU.CreateBuffer<float>(4);
        var kernel = new MultiAssignmentKernel(
            input.AsReadOnly(),
            bias.AsReadOnly(),
            expressionOutput.AsReadWrite(),
            copyOutput.AsReadWrite());

        GPU.Dispatch(kernel, 4);

        Assert.Equal([12, 24, 36, 48], expressionOutput.ToArray());
        Assert.Equal([1, 2, 3, 4], copyOutput.ToArray());
    }

    [Fact]
    public void ShaderInspectionBuildsTexture2DCopyFromTypedIr()
    {
        var glsl = ShaderInspection.GetGLSL<TextureCopyKernel>();

        Assert.Contains("image2D", glsl, StringComparison.Ordinal);
        Assert.Contains("imageLoad", glsl, StringComparison.Ordinal);
        Assert.Contains("imageStore", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void DispatchCopiesTexture2DThroughTypedEasyGpuIr()
    {
        var pixels = new[]
        {
            new Rgba32(1, 2, 3, 4),
            new Rgba32(5, 6, 7, 8),
            new Rgba32(9, 10, 11, 12),
            new Rgba32(13, 14, 15, 16)
        };
        using var input = GPU.CreateTexture2D<Rgba32, Rgba32>(2, 2, PixelFormat.Rgba8, TextureAccess.ReadOnly);
        using var output = GPU.CreateTexture2D<Rgba32, Rgba32>(2, 2, PixelFormat.Rgba8, TextureAccess.ReadWrite);
        input.Upload(pixels);
        var kernel = new TextureCopyKernel(input.AsReadOnly(), output.AsReadWrite());

        var path = DispatchAndGetPath(kernel, new GpuDispatchSize(2, 2, 1));

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        var readback = new Rgba32[4];
        output.Read(readback);
        Assert.Equal(pixels, readback);
    }

    [Fact]
    public void DispatchCopiesR8TextureThroughTypedEasyGpuIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            byte[] pixels = [64, 128, 192, 255];
            using var input = GPU.CreateTexture2D<byte, float4>(2, 2, PixelFormat.R8, TextureAccess.ReadOnly);
            using var output = GPU.CreateBuffer<float>(4);
            input.Upload(pixels);

            var path = DispatchAndGetPath(
                new TextureR8LoadKernel(input.AsReadOnly(), output.AsReadWrite()),
                new GpuDispatchSize(2, 2, 1));

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            AssertNear(pixels.Select(pixel => pixel / 255.0f).ToArray(), output.ToArray(), 1e-4f);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchCopiesRg8TextureThroughTypedEasyGpuIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            Rg8[] pixels =
            [
                new(32, 64),
                new(96, 128),
                new(160, 192),
                new(224, 255)
            ];
            using var input = GPU.CreateTexture2D<Rg8, float4>(2, 2, PixelFormat.Rg8, TextureAccess.ReadOnly);
            using var output = GPU.CreateBuffer<float2>(4);
            input.Upload(pixels);

            var path = DispatchAndGetPath(
                new TextureRg8LoadKernel(input.AsReadOnly(), output.AsReadWrite()),
                new GpuDispatchSize(2, 2, 1));

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            AssertFloat2Near(
                pixels.Select(pixel => new float2(pixel.R / 255.0f, pixel.G / 255.0f)).ToArray(),
                output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchCopiesRgba8TextureThroughTypedEasyGpuIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            Rgba32[] pixels =
            [
                new(1, 2, 3, 4),
                new(5, 6, 7, 8),
                new(9, 10, 11, 12),
                new(13, 14, 15, 16)
            ];
            using var input = GPU.CreateTexture2D<Rgba32, Rgba32>(2, 2, PixelFormat.Rgba8, TextureAccess.ReadOnly);
            using var output = GPU.CreateTexture2D<Rgba32, Rgba32>(2, 2, PixelFormat.Rgba8, TextureAccess.ReadWrite);
            input.Upload(pixels);

            var path = DispatchAndGetPath(
                new TextureCopyKernel(input.AsReadOnly(), output.AsReadWrite()),
                new GpuDispatchSize(2, 2, 1));

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            var readback = new Rgba32[4];
            output.Read(readback);
            Assert.Equal(pixels, readback);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchCopiesR32FloatTextureThroughTypedEasyGpuIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            float[] pixels = [0.25f, 0.5f, 0.75f, 1.0f];
            using var input = GPU.CreateTexture2D<float, float4>(2, 2, PixelFormat.R32Float, TextureAccess.ReadOnly);
            using var output = GPU.CreateBuffer<float>(4);
            input.Upload(pixels);

            var path = DispatchAndGetPath(
                new TextureR32FloatLoadKernel(input.AsReadOnly(), output.AsReadWrite()),
                new GpuDispatchSize(2, 2, 1));

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            AssertNear(pixels, output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchCopiesRgba32FloatTextureThroughTypedEasyGpuIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            float4[] pixels =
            [
                new(1, 2, 3, 4),
                new(5, 6, 7, 8),
                new(9, 10, 11, 12),
                new(13, 14, 15, 16)
            ];
            using var input = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadOnly);
            using var output = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadWrite);
            input.Upload(pixels);

            var path = DispatchAndGetPath(
                new TextureFloat4CopyKernel(input.AsReadOnly(), output.AsReadWrite()),
                new GpuDispatchSize(2, 2, 1));

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            var readback = new float4[4];
            output.Read(readback);
            AssertFloat4Near(pixels, readback);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchRejectsBgra8TextureWithActionableTypedEasyGpuError()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            Rgba32[] pixels = [new(1, 2, 3, 4)];
            using var input = GPU.CreateTexture2D<Rgba32, Rgba32>(1, 1, PixelFormat.Bgra8, TextureAccess.ReadOnly);
            using var output = GPU.CreateTexture2D<Rgba32, Rgba32>(1, 1, PixelFormat.Bgra8, TextureAccess.ReadWrite);
            input.Upload(pixels);

            var exception = Assert.Throws<FeatherNativeException>(() =>
                DispatchAndGetPath(new TextureCopyKernel(input.AsReadOnly(), output.AsReadWrite()), new GpuDispatchSize(1, 1, 1)));

            Assert.Equal(FeResult.ErrorUnsupported, exception.Result);
            Assert.Contains("Bgra8", exception.Message, StringComparison.Ordinal);
            Assert.Contains("typed texture bridge", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsTexture3DLoadStoreFromTypedIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<Texture3DCopyKernel>();

            Assert.Contains("image3D", glsl, StringComparison.Ordinal);
            Assert.Contains("imageLoad", glsl, StringComparison.Ordinal);
            Assert.Contains("imageStore", glsl, StringComparison.Ordinal);
            Assert.Contains("ivec3", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchReadsTexture3DThroughTypedEasyGpuIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            float4[] voxels =
            [
                new(1, 2, 3, 4),
                new(5, 6, 7, 8),
                new(9, 10, 11, 12),
                new(13, 14, 15, 16),
                new(17, 18, 19, 20),
                new(21, 22, 23, 24),
                new(25, 26, 27, 28),
                new(29, 30, 31, 32)
            ];
            using var input = GPU.CreateTexture3D<float4, float4>(2, 2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadOnly);
            using var output = GPU.CreateBuffer<float4>(8);
            input.Upload(voxels);

            var path = DispatchAndGetPath(
                new Texture3DReadKernel(input.AsReadOnly(), output.AsReadWrite()),
                new GpuDispatchSize(2, 2, 2));

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            AssertFloat4Near(voxels, output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchWritesTexture3DThroughTypedEasyGpuIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var output = GPU.CreateTexture3D<float4, float4>(2, 2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadWrite);

            var path = DispatchAndGetPath(
                new Texture3DWriteKernel(output.AsReadWrite()),
                new GpuDispatchSize(2, 2, 2));

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            var readback = new float4[8];
            output.Read(readback);
            AssertFloat4Near(Texture3DExpectedWriteValues(), readback);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchCopiesTexture3DThroughTypedEasyGpuIrWithDynamicInt3Indexing()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            var voxels = Texture3DExpectedWriteValues();
            using var input = GPU.CreateTexture3D<float4, float4>(2, 2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadOnly);
            using var output = GPU.CreateTexture3D<float4, float4>(2, 2, 2, PixelFormat.Rgba32Float, TextureAccess.ReadWrite);
            input.Upload(voxels);

            var path = DispatchAndGetPath(
                new Texture3DCopyKernel(input.AsReadOnly(), output.AsReadWrite()),
                new GpuDispatchSize(2, 2, 2));

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            var readback = new float4[8];
            output.Read(readback);
            AssertFloat4Near(voxels, readback);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsIntegerAtomicsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<AtomicOpsKernel>();

            Assert.Contains("atomicAdd", glsl, StringComparison.Ordinal);
            Assert.Matches(@"atomicAdd\([^,]+,\s*-\s*\(?1\)?", glsl);
            Assert.DoesNotContain("atomicSub", glsl, StringComparison.Ordinal);
            Assert.Contains("atomicMin", glsl, StringComparison.Ordinal);
            Assert.Contains("atomicMax", glsl, StringComparison.Ordinal);
            Assert.Contains("atomicAnd", glsl, StringComparison.Ordinal);
            Assert.Contains("atomicOr", glsl, StringComparison.Ordinal);
            Assert.Contains("atomicXor", glsl, StringComparison.Ordinal);
            Assert.Contains("atomicExchange", glsl, StringComparison.Ordinal);
            Assert.Contains("atomicCompSwap", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesAtomicAddFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<int>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<int>([0]);

            var path = DispatchAndGetPath(new AtomicAddKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([10], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesSharedMemoryAtomicAddWithDynamicIndexFromTypedIr()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<int>([3, 5, 7, 11]);
            using var output = GPU.CreateBuffer<int>(4);

            var path = DispatchAndGetPath(new SharedMemoryAtomicAddKernel(input.AsReadOnly(), output.AsReadWrite()), 4);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([3, 5, 7, 11], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsStructFieldReadsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<StructFieldReadKernel>();

            Assert.Contains("struct TypedScene", glsl, StringComparison.Ordinal);
            Assert.Contains("vec3 LightDir;", glsl, StringComparison.Ordinal);
            Assert.Contains("float Intensity;", glsl, StringComparison.Ordinal);
            Assert.Contains(".LightDir", glsl, StringComparison.Ordinal);
            Assert.Contains(".Intensity", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchReadsStructFieldsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<TypedScene>(
            [
                new TypedScene { LightDir = new float3(1, 2, 3), Intensity = 10 },
                new TypedScene { LightDir = new float3(4, 5, 6), Intensity = 20 }
            ]);
            using var output = GPU.CreateBuffer<float>(2);

            var path = DispatchAndGetPath(new StructFieldReadKernel(input.AsReadOnly(), output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([16, 35], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchWritesStructFieldsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var intensity = GPU.CreateBuffer<float>([2, 4]);
            using var output = GPU.CreateBuffer<MutableScene>(2);

            var path = DispatchAndGetPath(new StructFieldWriteKernel(intensity.AsReadOnly(), output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            var result = output.ToArray();
            Assert.Equal(new float3(1, 2, 3), result[0].LightDir);
            Assert.Equal(2, result[0].Intensity);
            Assert.Equal(new float3(2, 3, 4), result[1].LightDir);
            Assert.Equal(5, result[1].Intensity);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsNestedStructFieldAccessFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var readGlsl = ShaderInspection.GetGLSL<NestedStructFieldReadKernel>();
            var writeGlsl = ShaderInspection.GetGLSL<NestedStructFieldWriteKernel>();

            Assert.Contains("struct TypedScene", readGlsl, StringComparison.Ordinal);
            Assert.Contains("struct NestedScene", readGlsl, StringComparison.Ordinal);
            Assert.Contains(".Scene", readGlsl, StringComparison.Ordinal);
            Assert.Contains(".LightDir", readGlsl, StringComparison.Ordinal);
            Assert.Contains(".Intensity", readGlsl, StringComparison.Ordinal);
            Assert.Contains(".Scene", writeGlsl, StringComparison.Ordinal);
            Assert.Contains(".LightDir", writeGlsl, StringComparison.Ordinal);
            Assert.Contains(".Intensity", writeGlsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", readGlsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", writeGlsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchReadsAndWritesNestedStructFieldsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<NestedScene>(
            [
                new NestedScene
                {
                    Scene = new TypedScene { LightDir = new float3(1, 2, 3), Intensity = 4 },
                    Weight = new float4(10, 20, 30, 40)
                },
                new NestedScene
                {
                    Scene = new TypedScene { LightDir = new float3(5, 6, 7), Intensity = 8 },
                    Weight = new float4(1, 2, 3, 4)
                }
            ]);
            using var readOutput = GPU.CreateBuffer<float>(2);
            using var written = GPU.CreateBuffer<NestedScene>(2);

            var readPath = DispatchAndGetPath(new NestedStructFieldReadKernel(input.AsReadOnly(), readOutput.AsReadWrite()), 2);
            var writePath = DispatchAndGetPath(new NestedStructFieldWriteKernel(readOutput.AsReadOnly(), written.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, readPath);
            Assert.Equal(DispatchPath.TypedEasyGpu, writePath);
            Assert.Equal([20, 27], readOutput.ToArray());
            var result = written.ToArray();
            Assert.Equal(new float3(20, 21, 22), result[0].Scene.LightDir);
            Assert.Equal(25, result[0].Scene.Intensity);
            Assert.Equal(new float4(20, 10, 4, 1), result[0].Weight);
            Assert.Equal(new float3(28, 29, 30), result[1].Scene.LightDir);
            Assert.Equal(33, result[1].Scene.Intensity);
            Assert.Equal(new float4(28, 14, 5.6f, 1), result[1].Weight);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsCallableGpuStructReturnFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<CallableStructReturnKernel>();

            Assert.Contains("struct CallableHitInfo", glsl, StringComparison.Ordinal);
            Assert.Contains("CallableHitInfo(", glsl, StringComparison.Ordinal);
            Assert.Contains(".Closest", glsl, StringComparison.Ordinal);
            Assert.Contains(".Normal", glsl, StringComparison.Ordinal);
            Assert.Contains(".Color", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchReadsCallableGpuStructReturnFieldsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var input = GPU.CreateBuffer<float>([2, 5]);
            using var output = GPU.CreateBuffer<float>(2);

            var path = DispatchAndGetPath(new CallableStructReturnKernel(input.AsReadOnly(), output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([19, 31], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsGpuStructInstanceCallableFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<GpuStructInstanceCallableKernel>();

            Assert.Contains("struct ScaleBias", glsl, StringComparison.Ordinal);
            Assert.Contains("ScaleBias fe_this", glsl, StringComparison.Ordinal);
            Assert.Contains(".Scale", glsl, StringComparison.Ordinal);
            Assert.Contains(".Bias", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesGpuStructInstanceCallableFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var parameters = GPU.CreateBuffer<ScaleBias>(
            [
                new ScaleBias(2, 1),
                new ScaleBias(3, -4)
            ]);
            using var input = GPU.CreateBuffer<float>([5, 6]);
            using var output = GPU.CreateBuffer<float>(2);

            var path = DispatchAndGetPath(new GpuStructInstanceCallableKernel(
                parameters.AsReadOnly(),
                input.AsReadOnly(),
                output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([11, 14], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsMutatingGpuStructInstanceCallableWithInOutReceiverFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<GpuStructMutatingInstanceCallableKernel>();

            Assert.Contains("inout MutableCounter fe_this", glsl, StringComparison.Ordinal);
            Assert.Contains("inout MutableCounterNested fe_this", glsl, StringComparison.Ordinal);
            Assert.Contains("MutableCounter_Advance", glsl, StringComparison.Ordinal);
            Assert.Contains("MutableCounterNested_Accumulate", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesMutatingGpuStructInstanceCallableWithBufferAndLocalWritebackFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var counters = GPU.CreateBuffer<MutableCounter>(
            [
                new MutableCounter { Value = 10, Hits = 1, Nested = new MutableCounterNested { Inner = 5 } },
                new MutableCounter { Value = -3, Hits = 4, Nested = new MutableCounterNested { Inner = 1 } }
            ]);
            using var input = GPU.CreateBuffer<float>([2, 5]);
            using var output = GPU.CreateBuffer<float>(2);

            var path = DispatchAndGetPath(new GpuStructMutatingInstanceCallableKernel(
                counters.AsReadWrite(),
                input.AsReadOnly(),
                output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            AssertNear([24.5f, 20.5f], output.ToArray());

            var updated = counters.ToArray();
            Assert.Equal(2, updated[0].Hits);
            Assert.Equal(5, updated[1].Hits);
            AssertNear([13.5f, 3.5f], updated.Select(counter => counter.Value).ToArray());
            AssertNear([9.0f, 12.0f], updated.Select(counter => counter.Nested.Inner).ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsGenericInterfaceCallableMonomorphizationsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<GenericInterfaceShapeKernel>();

            Assert.Contains("MonoShapeOps_Eval_T_global__Feather_Integration_Tests_MonoSphere", glsl, StringComparison.Ordinal);
            Assert.Contains("MonoShapeOps_Eval_T_global__Feather_Integration_Tests_MonoPlane", glsl, StringComparison.Ordinal);
            Assert.Contains("MonoSphere_Sdf", glsl, StringComparison.Ordinal);
            Assert.Contains("MonoPlane_Sdf", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("IMonoShape", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("switch", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesGenericInterfaceCallableMonomorphizationsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var spheres = GPU.CreateBuffer<MonoSphere>(
            [
                new MonoSphere(2),
                new MonoSphere(1.5f)
            ]);
            using var planes = GPU.CreateBuffer<MonoPlane>(
            [
                new MonoPlane(1),
                new MonoPlane(-2)
            ]);
            using var points = GPU.CreateBuffer<float3>(
            [
                new float3(3, 4, 0),
                new float3(0, 0, 6)
            ]);
            using var output = GPU.CreateBuffer<float>(2);

            var path = DispatchAndGetPath(new GenericInterfaceShapeKernel(
                spheres.AsReadOnly(),
                planes.AsReadOnly(),
                points.AsReadOnly(),
                output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            AssertNear([8.0f, 2.5f], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsMutatingGenericInterfaceCallableMonomorphizationFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<GenericMutatingInterfaceShapeKernel>();

            Assert.Contains("MonoMutableOps_EvalAfterOffset_T_global__Feather_Integration_Tests_MonoMutableOffset", glsl, StringComparison.Ordinal);
            Assert.Contains("inout MonoMutableOffset fe_this", glsl, StringComparison.Ordinal);
            Assert.Contains("MonoMutableOffset_Offset", glsl, StringComparison.Ordinal);
            Assert.Contains("MonoMutableOffset_Measure", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("IMonoMutableShape", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("switch", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesMutatingGenericInterfaceCallableOnValueCopyFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var shapes = GPU.CreateBuffer<MonoMutableOffset>(
            [
                new MonoMutableOffset { Current = 10, Scale = 2 },
                new MonoMutableOffset { Current = 5, Scale = 6 }
            ]);
            using var deltas = GPU.CreateBuffer<float>([3, -2]);
            using var output = GPU.CreateBuffer<float>(2);

            var path = DispatchAndGetPath(new GenericMutatingInterfaceShapeKernel(
                shapes.AsReadOnly(),
                deltas.AsReadOnly(),
                output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            AssertNear([16, -7], output.ToArray());

            var unchanged = shapes.ToArray();
            AssertNear([10, 5], unchanged.Select(shape => shape.Current).ToArray());
            AssertNear([2, 6], unchanged.Select(shape => shape.Scale).ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsNestedGpuStructInstanceCallableGraphFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var glsl = ShaderInspection.GetGLSL<ComplexGpuStructInstanceCallableKernel>();

            Assert.Contains("struct ComplexScaleBias", glsl, StringComparison.Ordinal);
            Assert.Contains("struct CallableGain", glsl, StringComparison.Ordinal);
            Assert.Contains("ComplexScaleBias fe_this", glsl, StringComparison.Ordinal);
            Assert.Contains("CallableGain gain", glsl, StringComparison.Ordinal);
            Assert.Contains(".Mode", glsl, StringComparison.Ordinal);
            Assert.Contains(".Multiplier", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("ScaleOnly_int", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Unused", glsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void DispatchExecutesNestedGpuStructInstanceCallableGraphFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            using var curves = GPU.CreateBuffer<ComplexScaleBias>(
            [
                new ComplexScaleBias(2, 1, 0),
                new ComplexScaleBias(3, -2, 1)
            ]);
            using var gains = GPU.CreateBuffer<CallableGain>(
            [
                new CallableGain(10),
                new CallableGain(4)
            ]);
            using var input = GPU.CreateBuffer<float>([3, 5]);
            using var output = GPU.CreateBuffer<float>(2);

            var path = DispatchAndGetPath(new ComplexGpuStructInstanceCallableKernel(
                curves.AsReadOnly(),
                gains.AsReadOnly(),
                input.AsReadOnly(),
                output.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.Equal([70, 50], output.ToArray());
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ShaderInspectionBuildsGpuStructArrayAccessFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;

            var readGlsl = ShaderInspection.GetGLSL<StructArrayReadKernel>();
            var writeGlsl = ShaderInspection.GetGLSL<StructArrayWriteKernel>();

            Assert.Contains("vec3 Directions[4];", readGlsl, StringComparison.Ordinal);
            Assert.Contains(".Directions", readGlsl, StringComparison.Ordinal);
            Assert.Contains("[", readGlsl, StringComparison.Ordinal);
            Assert.Contains("InnerArrayScene Items[3];", writeGlsl, StringComparison.Ordinal);
            Assert.Contains(".Items", writeGlsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", readGlsl, StringComparison.Ordinal);
            Assert.DoesNotContain("Feather native stub", writeGlsl, StringComparison.Ordinal);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void GpuStructArrayLayoutPackUnpackAndBufferRoundTripUseStd430Stride()
    {
        var scene = new ArrayScene { Weight = 5 };
        scene.Directions[0] = new float3(1, 2, 3);
        scene.Directions[1] = new float3(4, 5, 6);
        scene.Directions[2] = new float3(7, 8, 9);
        scene.Directions[3] = new float3(10, 11, 12);
        var layout = GpuStructLayout<ArrayScene>();
        var field = layout.Fields.Single(field => field.Name == nameof(ArrayScene.Directions));
        var bytes = Enumerable.Repeat((byte)0xCD, layout.SizeInBytes).ToArray();

        GpuStructPack<ArrayScene>([scene], bytes);

        Assert.Equal(80, layout.SizeInBytes);
        Assert.Equal(16, layout.Alignment);
        Assert.Equal(0, field.Offset);
        Assert.Equal(64, field.SizeInBytes);
        Assert.Equal(16, field.Alignment);
        Assert.Equal(4, field.ArrayLength);
        Assert.Equal(16, field.ArrayStride);
        Assert.Equal(64, layout.Fields.Single(field => field.Name == nameof(ArrayScene.Weight)).Offset);
        Assert.Equal(1, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(0, 4)));
        Assert.Equal(3, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(8, 4)));
        Assert.All(bytes[12..16], value => Assert.Equal(0, value));
        Assert.Equal(4, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(16, 4)));
        Assert.Equal(7, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(32, 4)));
        Assert.Equal(10, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(48, 4)));
        Assert.Equal(5, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(64, 4)));

        var unpacked = new ArrayScene[1];
        GpuStructUnpack<ArrayScene>(bytes, unpacked);

        Assert.Equal(scene.Directions[0], unpacked[0].Directions[0]);
        Assert.Equal(scene.Directions[1], unpacked[0].Directions[1]);
        Assert.Equal(scene.Directions[2], unpacked[0].Directions[2]);
        Assert.Equal(scene.Directions[3], unpacked[0].Directions[3]);
        Assert.Equal(scene.Weight, unpacked[0].Weight);

        using var buffer = GPU.CreateBuffer<ArrayScene>([scene]);
        Assert.Equal(scene.Directions[2], buffer.ToArray()[0].Directions[2]);
    }

    [Fact]
    public void DispatchReadsAndWritesGpuStructArrayFieldsFromTypedIrWhenLegacySectionsAreRemoved()
    {
        try
        {
            GpuKernel.IrTransformForTesting = StripToTypedIrOnly;
            var first = new ArrayScene { Weight = 5 };
            first.Directions[0] = new float3(1, 2, 3);
            first.Directions[1] = new float3(4, 5, 6);
            first.Directions[2] = new float3(7, 8, 9);
            first.Directions[3] = new float3(10, 11, 12);
            var second = new ArrayScene { Weight = 20 };
            second.Directions[0] = new float3(2, 4, 6);
            second.Directions[1] = new float3(8, 10, 12);
            second.Directions[2] = new float3(14, 16, 18);
            second.Directions[3] = new float3(20, 22, 24);
            using var input = GPU.CreateBuffer<ArrayScene>([first, second]);
            using var output = GPU.CreateBuffer<float>(2);
            using var written = GPU.CreateBuffer<NestedArrayScene>(2);

            var readPath = DispatchAndGetPath(new StructArrayReadKernel(input.AsReadOnly(), output.AsReadWrite()), 2);
            var writePath = DispatchAndGetPath(new StructArrayWriteKernel(output.AsReadOnly(), written.AsReadWrite()), 2);

            Assert.Equal(DispatchPath.TypedEasyGpu, readPath);
            Assert.Equal(DispatchPath.TypedEasyGpu, writePath);
            Assert.Equal([20, 68], output.ToArray());
            var result = written.ToArray();
            Assert.Equal(20, result[0].Items[0].Value);
            Assert.Equal(21, result[0].Items[1].Value);
            Assert.Equal(22, result[0].Items[2].Value);
            Assert.Equal(40, result[0].Weight);
            Assert.Equal(69, result[1].Items[0].Value);
            Assert.Equal(70, result[1].Items[1].Value);
            Assert.Equal(71, result[1].Items[2].Value);
            Assert.Equal(138, result[1].Weight);
        }
        finally
        {
            GpuKernel.IrTransformForTesting = null;
        }
    }

    [Fact]
    public void ProfilerRecordsGeneratedComputeDispatch()
    {
        try
        {
            GpuProfiler.SetEnabled(true);
            GpuProfiler.Clear();

            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);
            var kernel = new CopyKernel(input.AsReadOnly(), output.AsReadWrite());

            GPU.Dispatch(kernel, 4);

            var query = GpuProfiler.Query(nameof(CopyKernel));
            Assert.Equal(1UL, query.Count);
            Assert.True(query.TotalTimeMs >= 0.0);
            Assert.True(query.MaxTimeMs >= query.MinTimeMs);
            Assert.True(GpuProfiler.GetTotalTimeMs() >= query.TotalTimeMs);
            Assert.Contains(nameof(CopyKernel), GpuProfiler.GetFormattedReport(), StringComparison.Ordinal);
        }
        finally
        {
            GpuProfiler.Clear();
            GpuProfiler.SetEnabled(false);
        }
    }

    [Fact]
    public void ProfilerDoesNotRecordWhenDisabled()
    {
        try
        {
            GpuProfiler.SetEnabled(false);
            GpuProfiler.Clear();

            using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
            using var output = GPU.CreateBuffer<float>(4);
            var kernel = new CopyKernel(input.AsReadOnly(), output.AsReadWrite());

            GPU.Dispatch(kernel, 4);

            var query = GpuProfiler.Query(nameof(CopyKernel));
            Assert.Equal(0UL, query.Count);
            Assert.Equal(0.0, GpuProfiler.GetTotalTimeMs());
        }
        finally
        {
            GpuProfiler.Clear();
            GpuProfiler.SetEnabled(false);
        }
    }

    [Fact]
    public void GpuStructPackAndUnpackUseGeneratedStd430Layout()
    {
        var values = new[] { new PackedScene { LightDir = new float3(1, 2, 3), Intensity = 4 } };
        var layout = PackedSceneLayout();
        var bytes = new byte[layout.SizeInBytes];

        PackPackedScene(values, bytes);

        Assert.Equal(16, layout.SizeInBytes);
        Assert.Equal(16, layout.Alignment);
        Assert.Equal(0, layout.Fields.Single(field => field.Name == nameof(PackedScene.LightDir)).Offset);
        Assert.Equal(12, layout.Fields.Single(field => field.Name == nameof(PackedScene.Intensity)).Offset);

        var unpacked = new PackedScene[1];
        UnpackPackedScene(bytes, unpacked);

        Assert.Equal(values[0].LightDir, unpacked[0].LightDir);
        Assert.Equal(values[0].Intensity, unpacked[0].Intensity);
    }

    [Fact]
    public void NestedGpuStructLayoutUsesEasyGpuStd430Offsets()
    {
        var layout = GpuStructLayout<NestedScene>();
        var inner = GpuStructLayout<TypedScene>();

        Assert.Equal(16, inner.SizeInBytes);
        Assert.Equal(16, inner.Alignment);
        Assert.Equal(32, layout.SizeInBytes);
        Assert.Equal(16, layout.Alignment);
        Assert.Equal(0, layout.Fields.Single(field => field.Name == nameof(NestedScene.Scene)).Offset);
        Assert.Equal(16, layout.Fields.Single(field => field.Name == nameof(NestedScene.Weight)).Offset);
    }

    [Fact]
    public void MatrixGpuStructLayoutUsesEasyGpuStd430OffsetsAndPadding()
    {
        var value = new MatrixScene
        {
            Weight = 5,
            Transform = new float2x2(new float2(1, 2), new float2(3, 4)),
            Bias = 7
        };
        var layout = GpuStructLayout<MatrixScene>();
        var bytes = Enumerable.Repeat((byte)0xCD, layout.SizeInBytes).ToArray();

        GpuStructPack<MatrixScene>([value], bytes);

        Assert.Equal(32, layout.SizeInBytes);
        Assert.Equal(8, layout.Alignment);
        Assert.Equal(0, layout.Fields.Single(field => field.Name == nameof(MatrixScene.Weight)).Offset);
        Assert.Equal(4, layout.Fields.Single(field => field.Name == nameof(MatrixScene.Weight)).SizeInBytes);
        Assert.Equal(8, layout.Fields.Single(field => field.Name == nameof(MatrixScene.Transform)).Offset);
        Assert.Equal(16, layout.Fields.Single(field => field.Name == nameof(MatrixScene.Transform)).SizeInBytes);
        Assert.Equal(8, layout.Fields.Single(field => field.Name == nameof(MatrixScene.Transform)).Alignment);
        Assert.Equal(24, layout.Fields.Single(field => field.Name == nameof(MatrixScene.Bias)).Offset);
        Assert.Equal(4, layout.Fields.Single(field => field.Name == nameof(MatrixScene.Bias)).SizeInBytes);

        Assert.Equal(5, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(0, 4)));
        Assert.All(bytes[4..8], value => Assert.Equal(0, value));
        Assert.Equal(1, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(8, 4)));
        Assert.Equal(2, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(12, 4)));
        Assert.Equal(3, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(16, 4)));
        Assert.Equal(4, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(20, 4)));
        Assert.Equal(7, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(24, 4)));
        Assert.All(bytes[28..32], value => Assert.Equal(0, value));

        var unpacked = new MatrixScene[1];
        GpuStructUnpack<MatrixScene>(bytes, unpacked);

        Assert.Equal(value.Weight, unpacked[0].Weight);
        Assert.Equal(value.Transform, unpacked[0].Transform);
        Assert.Equal(value.Bias, unpacked[0].Bias);
    }

    [Fact]
    public void BoolGpuValueLayoutsUseThirtyTwoBitShaderSlots()
    {
        Assert.Equal(4, GpuValueLayout<bool>.BufferElementStride);
        Assert.Equal(4, GpuValueLayout<bool>.FieldSizeInBytes);
        Assert.Equal(4, GpuValueLayout<bool>.Alignment);
        Assert.True(GpuValueLayout<bool>.RequiresBufferRepacking);

        Assert.Equal(8, GpuValueLayout<bool2>.BufferElementStride);
        Assert.Equal(8, GpuValueLayout<bool2>.FieldSizeInBytes);
        Assert.Equal(8, GpuValueLayout<bool2>.Alignment);

        Assert.Equal(16, GpuValueLayout<bool3>.BufferElementStride);
        Assert.Equal(12, GpuValueLayout<bool3>.FieldSizeInBytes);
        Assert.Equal(16, GpuValueLayout<bool3>.Alignment);

        Assert.Equal(16, GpuValueLayout<bool4>.BufferElementStride);
        Assert.Equal(16, GpuValueLayout<bool4>.FieldSizeInBytes);
        Assert.Equal(16, GpuValueLayout<bool4>.Alignment);

        var bools = Enumerable.Repeat((byte)0xCD, GpuValueLayout<bool>.BufferElementStride * 3).ToArray();
        GpuValueLayout<bool>.PackBuffer([true, false, true], bools);

        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(bools.AsSpan(0, 4)));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(bools.AsSpan(4, 4)));
        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(bools.AsSpan(8, 4)));

        var unpackedBools = new bool[3];
        GpuValueLayout<bool>.UnpackBuffer(bools, unpackedBools);
        Assert.Equal([true, false, true], unpackedBools);

        var vectors = Enumerable.Repeat((byte)0xCD, GpuValueLayout<bool3>.BufferElementStride * 2).ToArray();
        GpuValueLayout<bool3>.PackBuffer([new bool3(true, false, true), new bool3(false, true, false)], vectors);

        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(vectors.AsSpan(0, 4)));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(vectors.AsSpan(4, 4)));
        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(vectors.AsSpan(8, 4)));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(vectors.AsSpan(12, 4)));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(vectors.AsSpan(16, 4)));
        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(vectors.AsSpan(20, 4)));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(vectors.AsSpan(24, 4)));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(vectors.AsSpan(28, 4)));

        var unpackedVectors = new bool3[2];
        GpuValueLayout<bool3>.UnpackBuffer(vectors, unpackedVectors);
        Assert.Equal([new bool3(true, false, true), new bool3(false, true, false)], unpackedVectors);
    }

    [Fact]
    public void BoolVectorGpuStructLayoutUsesThirtyTwoBitSlotsAndPadding()
    {
        var value = new BoolVectorScene
        {
            Enabled = true,
            Mask = new bool3(true, false, true),
            Weight = 11
        };
        var layout = GpuStructLayout<BoolVectorScene>();
        var bytes = Enumerable.Repeat((byte)0xCD, layout.SizeInBytes).ToArray();

        GpuStructPack<BoolVectorScene>([value], bytes);

        Assert.Equal(32, layout.SizeInBytes);
        Assert.Equal(16, layout.Alignment);
        Assert.Equal(0, layout.Fields.Single(field => field.Name == nameof(BoolVectorScene.Enabled)).Offset);
        Assert.Equal(4, layout.Fields.Single(field => field.Name == nameof(BoolVectorScene.Enabled)).SizeInBytes);
        Assert.Equal(16, layout.Fields.Single(field => field.Name == nameof(BoolVectorScene.Mask)).Offset);
        Assert.Equal(12, layout.Fields.Single(field => field.Name == nameof(BoolVectorScene.Mask)).SizeInBytes);
        Assert.Equal(16, layout.Fields.Single(field => field.Name == nameof(BoolVectorScene.Mask)).Alignment);
        Assert.Equal(28, layout.Fields.Single(field => field.Name == nameof(BoolVectorScene.Weight)).Offset);

        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(0, 4)));
        Assert.All(bytes[4..16], value => Assert.Equal(0, value));
        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(16, 4)));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(20, 4)));
        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(24, 4)));
        Assert.Equal(11, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(28, 4)));

        var unpacked = new BoolVectorScene[1];
        GpuStructUnpack<BoolVectorScene>(bytes, unpacked);

        Assert.Equal(value.Enabled, unpacked[0].Enabled);
        Assert.Equal(value.Mask, unpacked[0].Mask);
        Assert.Equal(value.Weight, unpacked[0].Weight);
    }

    [Fact]
    public void ManagedValueLayoutsMatchEasyGpuStd430Contracts()
    {
        Assert.Equal(12, GpuValueLayout<float3>.CpuSizeInBytes);
        Assert.Equal(16, GpuValueLayout<float3>.BufferElementStride);
        Assert.Equal(12, GpuValueLayout<float3>.FieldSizeInBytes);
        Assert.Equal(16, GpuValueLayout<float3>.Alignment);
        Assert.True(GpuValueLayout<float3>.RequiresBufferRepacking);

        Assert.Equal(16, GpuValueLayout<float2x2>.CpuSizeInBytes);
        Assert.Equal(16, GpuValueLayout<float2x2>.BufferElementStride);
        Assert.Equal(16, GpuValueLayout<float2x2>.FieldSizeInBytes);
        Assert.Equal(8, GpuValueLayout<float2x2>.Alignment);
        Assert.False(GpuValueLayout<float2x2>.RequiresBufferRepacking);

        Assert.Equal(36, GpuValueLayout<float3x3>.CpuSizeInBytes);
        Assert.Equal(48, GpuValueLayout<float3x3>.BufferElementStride);
        Assert.Equal(48, GpuValueLayout<float3x3>.FieldSizeInBytes);
        Assert.Equal(16, GpuValueLayout<float3x3>.Alignment);
        Assert.True(GpuValueLayout<float3x3>.RequiresBufferRepacking);

        Assert.Equal(64, GpuValueLayout<float4x4>.CpuSizeInBytes);
        Assert.Equal(64, GpuValueLayout<float4x4>.BufferElementStride);
        Assert.Equal(64, GpuValueLayout<float4x4>.FieldSizeInBytes);
        Assert.Equal(16, GpuValueLayout<float4x4>.Alignment);
        Assert.False(GpuValueLayout<float4x4>.RequiresBufferRepacking);
    }

    [Fact]
    public void MatrixValueLayoutsPackColumnsWithSixteenByteStride()
    {
        var matrix2 = new float2x2(
            new float2(1, 2),
            new float2(3, 4));
        var matrix2Bytes = Enumerable.Repeat((byte)0xCD, GpuValueLayout<float2x2>.FieldSizeInBytes).ToArray();

        GpuValueLayout<float2x2>.PackValue(in matrix2, matrix2Bytes);

        Assert.Equal(1, BinaryPrimitives.ReadSingleLittleEndian(matrix2Bytes.AsSpan(0, 4)));
        Assert.Equal(2, BinaryPrimitives.ReadSingleLittleEndian(matrix2Bytes.AsSpan(4, 4)));
        Assert.Equal(3, BinaryPrimitives.ReadSingleLittleEndian(matrix2Bytes.AsSpan(8, 4)));
        Assert.Equal(4, BinaryPrimitives.ReadSingleLittleEndian(matrix2Bytes.AsSpan(12, 4)));
        Assert.Equal(matrix2, GpuValueLayout<float2x2>.UnpackValue(matrix2Bytes));

        var matrix2Array = new[]
        {
            matrix2,
            new float2x2(
                new float2(5, 6),
                new float2(7, 8))
        };
        var matrix2BufferBytes = Enumerable.Repeat((byte)0xCD, GpuValueLayout<float2x2>.BufferElementStride * matrix2Array.Length).ToArray();

        GpuValueLayout<float2x2>.PackBuffer(matrix2Array, matrix2BufferBytes);

        Assert.Equal(5, BinaryPrimitives.ReadSingleLittleEndian(matrix2BufferBytes.AsSpan(16, 4)));
        Assert.Equal(6, BinaryPrimitives.ReadSingleLittleEndian(matrix2BufferBytes.AsSpan(20, 4)));
        Assert.Equal(7, BinaryPrimitives.ReadSingleLittleEndian(matrix2BufferBytes.AsSpan(24, 4)));
        Assert.Equal(8, BinaryPrimitives.ReadSingleLittleEndian(matrix2BufferBytes.AsSpan(28, 4)));
        var unpackedMatrix2 = new float2x2[2];
        GpuValueLayout<float2x2>.UnpackBuffer(matrix2BufferBytes, unpackedMatrix2);
        Assert.Equal(matrix2Array, unpackedMatrix2);

        var matrix3 = new float3x3(
            new float3(1, 2, 3),
            new float3(4, 5, 6),
            new float3(7, 8, 9));
        var matrix3Bytes = Enumerable.Repeat((byte)0xCD, GpuValueLayout<float3x3>.FieldSizeInBytes).ToArray();

        GpuValueLayout<float3x3>.PackValue(in matrix3, matrix3Bytes);

        Assert.Equal(1, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(0, 4)));
        Assert.Equal(2, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(4, 4)));
        Assert.Equal(3, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(8, 4)));
        Assert.All(matrix3Bytes[12..16], value => Assert.Equal(0, value));
        Assert.Equal(4, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(16, 4)));
        Assert.Equal(5, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(20, 4)));
        Assert.Equal(6, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(24, 4)));
        Assert.All(matrix3Bytes[28..32], value => Assert.Equal(0, value));
        Assert.Equal(7, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(32, 4)));
        Assert.Equal(8, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(36, 4)));
        Assert.Equal(9, BinaryPrimitives.ReadSingleLittleEndian(matrix3Bytes.AsSpan(40, 4)));
        Assert.All(matrix3Bytes[44..48], value => Assert.Equal(0, value));
        Assert.Equal(matrix3, GpuValueLayout<float3x3>.UnpackValue(matrix3Bytes));

        var matrix4 = new float4x4(
            new float4(1, 2, 3, 4),
            new float4(5, 6, 7, 8),
            new float4(9, 10, 11, 12),
            new float4(13, 14, 15, 16));
        var matrix4Bytes = Enumerable.Repeat((byte)0xCD, GpuValueLayout<float4x4>.FieldSizeInBytes).ToArray();

        GpuValueLayout<float4x4>.PackValue(in matrix4, matrix4Bytes);

        Assert.Equal(1, BinaryPrimitives.ReadSingleLittleEndian(matrix4Bytes.AsSpan(0, 4)));
        Assert.Equal(5, BinaryPrimitives.ReadSingleLittleEndian(matrix4Bytes.AsSpan(16, 4)));
        Assert.Equal(9, BinaryPrimitives.ReadSingleLittleEndian(matrix4Bytes.AsSpan(32, 4)));
        Assert.Equal(13, BinaryPrimitives.ReadSingleLittleEndian(matrix4Bytes.AsSpan(48, 4)));
        Assert.Equal(matrix4, GpuValueLayout<float4x4>.UnpackValue(matrix4Bytes));

        var matrices = new[]
        {
            matrix3,
            new float3x3(
                new float3(10, 11, 12),
                new float3(13, 14, 15),
                new float3(16, 17, 18))
        };
        var bufferBytes = Enumerable.Repeat((byte)0xCD, GpuValueLayout<float3x3>.BufferElementStride * matrices.Length).ToArray();

        GpuValueLayout<float3x3>.PackBuffer(matrices, bufferBytes);

        Assert.Equal(10, BinaryPrimitives.ReadSingleLittleEndian(bufferBytes.AsSpan(48, 4)));
        Assert.Equal(13, BinaryPrimitives.ReadSingleLittleEndian(bufferBytes.AsSpan(64, 4)));
        Assert.Equal(16, BinaryPrimitives.ReadSingleLittleEndian(bufferBytes.AsSpan(80, 4)));
        var unpacked = new float3x3[2];
        GpuValueLayout<float3x3>.UnpackBuffer(bufferBytes, unpacked);
        Assert.Equal(matrices, unpacked);
    }

    private static GpuStructLayout PackedSceneLayout()
        => GpuStructLayout<PackedScene>();

    private static void PackPackedScene(ReadOnlySpan<PackedScene> source, Span<byte> destination)
        => GpuStructPack<PackedScene>(source, destination);

    private static void UnpackPackedScene(ReadOnlySpan<byte> source, Span<PackedScene> destination)
        => GpuStructUnpack<PackedScene>(source, destination);

    private static GpuStructLayout GpuStructLayout<T>()
        where T : unmanaged, IGpuStruct<T>
        => T.Layout;

    private static void GpuStructPack<T>(ReadOnlySpan<T> source, Span<byte> destination)
        where T : unmanaged, IGpuStruct<T>
        => T.Pack(source, destination);

    private static void GpuStructUnpack<T>(ReadOnlySpan<byte> source, Span<T> destination)
        where T : unmanaged, IGpuStruct<T>
        => T.Unpack(source, destination);

    private static void AssertNear(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            Assert.InRange(MathF.Abs(actual[i] - expected[i]), 0, tolerance);
        }
    }

    private static void AssertFloat2Near(IReadOnlyList<float2> expected, IReadOnlyList<float2> actual, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            Assert.InRange(MathF.Abs(actual[i].X - expected[i].X), 0, tolerance);
            Assert.InRange(MathF.Abs(actual[i].Y - expected[i].Y), 0, tolerance);
        }
    }

    private static void AssertFloat4Near(IReadOnlyList<float4> expected, IReadOnlyList<float4> actual, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            Assert.InRange(MathF.Abs(actual[i].X - expected[i].X), 0, tolerance);
            Assert.InRange(MathF.Abs(actual[i].Y - expected[i].Y), 0, tolerance);
            Assert.InRange(MathF.Abs(actual[i].Z - expected[i].Z), 0, tolerance);
            Assert.InRange(MathF.Abs(actual[i].W - expected[i].W), 0, tolerance);
        }
    }

    private static byte[] RemoveAssignmentCompatibilityOperands(ReadOnlySpan<byte> ir)
    {
        var bytes = ir.ToArray();
        var resourceCount = BitConverter.ToUInt32(bytes, 28);
        var instructionCount = BitConverter.ToUInt32(bytes, 36);
        var instructionOffset = checked(44 + ((int)resourceCount * 15));
        for (var i = 0; i < instructionCount; i++)
        {
            var offset = instructionOffset + (i * 8);
            if (bytes[offset] == 2 && bytes[offset + 1] == 2)
            {
                // Keep the assignment instruction but remove the legacy ASSIGN1 operand.
                // The native fallback must use the structured v1.1 assignment section instead.
                bytes[offset + 1] = 0;
                BitConverter.GetBytes(0u).CopyTo(bytes, offset + 4);
            }
        }

        return bytes;
    }

    private static byte[] StripSection7ForNativeReferenceFallback(ReadOnlySpan<byte> ir)
    {
        var resourceCount = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(28, sizeof(uint)));
        var instructionCount = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(36, sizeof(uint)));
        var stringByteLength = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(40, sizeof(uint)));
        var resourceBytes = checked((int)resourceCount * 15);
        var instructionBytes = checked((int)instructionCount * 8);
        var sectionHeaderOffset = checked(44 + resourceBytes + instructionBytes);
        var sectionCount = BinaryPrimitives.ReadUInt16LittleEndian(ir.Slice(10, sizeof(ushort)));
        var sectionPayloadOffset = checked(sectionHeaderOffset + (sectionCount * 8));
        var stringOffset = checked(ir.Length - (int)stringByteLength);

        using var stream = new MemoryStream();
        stream.Write(ir.Slice(0, 10));
        Span<byte> sectionCountBytes = stackalloc byte[2];
        BinaryPrimitives.WriteUInt16LittleEndian(sectionCountBytes, checked((ushort)(sectionCount - 1)));
        stream.Write(sectionCountBytes);
        stream.Write(ir.Slice(12, checked(sectionHeaderOffset - 12)));

        var payloadCursor = sectionPayloadOffset;
        Span<byte> sectionHeader = stackalloc byte[8];
        for (var i = 0; i < sectionCount; i++)
        {
            var header = sectionHeaderOffset + (i * 8);
            var kind = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(header, sizeof(uint)));
            var length = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(header + 4, sizeof(uint))));
            if (kind != 7)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(sectionHeader, kind);
                BinaryPrimitives.WriteUInt32LittleEndian(sectionHeader.Slice(4), checked((uint)length));
                stream.Write(sectionHeader);
            }

            payloadCursor = checked(payloadCursor + length);
        }

        Assert.Equal(stringOffset, payloadCursor);

        payloadCursor = sectionPayloadOffset;
        var foundTypedSection = false;
        for (var i = 0; i < sectionCount; i++)
        {
            var header = sectionHeaderOffset + (i * 8);
            var kind = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(header, sizeof(uint)));
            var length = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(header + 4, sizeof(uint))));
            if (kind == 7)
            {
                foundTypedSection = true;
            }
            else
            {
                stream.Write(ir.Slice(payloadCursor, length));
            }

            payloadCursor = checked(payloadCursor + length);
        }

        Assert.True(foundTypedSection, "Generated IR must contain typed section 7 before fallback stripping.");
        stream.Write(ir.Slice(stringOffset, (int)stringByteLength));
        return stream.ToArray();
    }

    private static byte[] StripToTypedIrOnly(ReadOnlySpan<byte> ir)
    {
        var resourceCount = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(28, sizeof(uint)));
        var instructionCount = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(36, sizeof(uint)));
        var stringByteLength = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(40, sizeof(uint)));
        var resourceBytes = checked((int)resourceCount * 15);
        var instructionBytes = checked((int)instructionCount * 8);
        var instructionOffset = checked(44 + resourceBytes);
        var sectionHeaderOffset = checked(instructionOffset + instructionBytes);
        var sectionCount = BinaryPrimitives.ReadUInt16LittleEndian(ir.Slice(10, sizeof(ushort)));
        var sectionPayloadOffset = checked(sectionHeaderOffset + (sectionCount * 8));
        var stringOffset = checked(ir.Length - (int)stringByteLength);

        var typedSectionOffset = -1;
        var typedSectionLength = 0;
        var payloadCursor = sectionPayloadOffset;
        for (var i = 0; i < sectionCount; i++)
        {
            var header = sectionHeaderOffset + (i * 8);
            var kind = BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(header, sizeof(uint)));
            var length = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ir.Slice(header + 4, sizeof(uint))));
            if (kind == 7)
            {
                typedSectionOffset = payloadCursor;
                typedSectionLength = length;
            }

            payloadCursor = checked(payloadCursor + length);
        }

        Assert.True(typedSectionOffset >= 0, "Generated IR must contain typed section 7.");
        Assert.Equal(stringOffset, payloadCursor);

        using var stream = new MemoryStream();
        stream.Write(ir.Slice(0, 10));
        Span<byte> sectionCountBytes = stackalloc byte[2];
        BinaryPrimitives.WriteUInt16LittleEndian(sectionCountBytes, 1);
        stream.Write(sectionCountBytes);
        stream.Write(ir.Slice(12, checked(sectionHeaderOffset - 12)));

        var bytes = stream.ToArray();
        for (var i = 0; i < instructionCount; i++)
        {
            var offset = instructionOffset + (i * 8);
            if (bytes[offset] == 2)
            {
                bytes[offset + 1] = 0;
                BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset + 4, sizeof(uint)), 0);
            }
        }

        stream.SetLength(0);
        stream.Write(bytes);

        Span<byte> sectionHeader = stackalloc byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(sectionHeader, 7);
        BinaryPrimitives.WriteUInt32LittleEndian(sectionHeader.Slice(4), checked((uint)typedSectionLength));
        stream.Write(sectionHeader);
        stream.Write(ir.Slice(typedSectionOffset, typedSectionLength));
        stream.Write(ir.Slice(stringOffset, (int)stringByteLength));
        return stream.ToArray();
    }

    private static byte[] CorruptTypedOnlyResourceName(ReadOnlySpan<byte> ir)
    {
        var bytes = StripToTypedIrOnly(ir);
        var resourceCount = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(28, sizeof(uint)));
        var instructionCount = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(36, sizeof(uint)));
        var resourceBytes = checked((int)resourceCount * 15);
        var instructionBytes = checked((int)instructionCount * 8);
        var sectionHeaderOffset = checked(44 + resourceBytes + instructionBytes);
        var typedSectionLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(sectionHeaderOffset + 4, sizeof(uint))));
        var typedPayload = bytes.AsSpan(sectionHeaderOffset + 8, typedSectionLength);

        Assert.Equal("FTIR"u8.ToArray(), typedPayload[..4].ToArray());
        var stringOffset = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(typedPayload.Slice(96, sizeof(uint))));
        var stringLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(typedPayload.Slice(100, sizeof(uint))));
        Assert.InRange(stringOffset + stringLength, 0, typedPayload.Length);

        var cursor = stringOffset;
        var count = BinaryPrimitives.ReadUInt32LittleEndian(typedPayload.Slice(cursor, sizeof(uint)));
        cursor += sizeof(uint);
        for (var i = 0; i < count; i++)
        {
            var length = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(typedPayload.Slice(cursor, sizeof(uint))));
            cursor += sizeof(uint);
            var value = typedPayload.Slice(cursor, length);
            if (value.SequenceEqual("input"u8))
            {
                "ghost"u8.CopyTo(value);
                return bytes;
            }

            cursor += length;
        }

        throw new InvalidOperationException("Expected section 7 string table to contain 'input'.");
    }

    private static DispatchPath DispatchAndGetPath<TKernel>(TKernel kernel, int x, bool wait = true)
        where TKernel : struct, IGeneratedKernel<TKernel>
        => DispatchAndGetPath(kernel, new GpuDispatchSize(x, 1, 1), wait);

    private static DispatchPath DispatchAndGetPath<TKernel>(TKernel kernel, GpuDispatchSize size, bool wait = true)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        using var gpuKernel = GpuKernel.Create<TKernel>(GPU.Context);
        GpuKernel.Dispatch(GPU.Context, gpuKernel, kernel, size, wait);
        return gpuKernel.LastDispatchPath;
    }

    private static float4[] Texture3DExpectedWriteValues()
        =>
        [
            new(0, 0, 0, 1),
            new(1, 0, 1, 1),
            new(0, 1, 10, 1),
            new(1, 1, 11, 1),
            new(0, 0, 100, 1),
            new(1, 0, 101, 1),
            new(0, 1, 110, 1),
            new(1, 1, 111, 1)
        ];

    internal static byte[] BuildRawSwizzleWriteIrForTesting(string components)
        => BuildRawSwizzleWriteIr(components);

    private static byte[] BuildRawSwizzleWriteIr(string components)
    {
        var typedSection = BuildRawSwizzleWriteTypedSection(components);
        var strings = BuildStringTable(["RawSwizzleWriteKernel", "output", "Feather.Math.float4"]);

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write("FEIR"u8);
        writer.Write((ushort)1);
        writer.Write((ushort)1);
        writer.Write((byte)1);
        writer.Write((byte)1);
        writer.Write((ushort)1);
        writer.Write(1);
        writer.Write(1);
        writer.Write(1);
        writer.Write((uint)0);
        writer.Write((uint)1);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)strings.Length);

        writer.Write((uint)0);
        writer.Write((byte)1);
        writer.Write((byte)3);
        writer.Write((byte)0);
        writer.Write((uint)1);
        writer.Write((uint)2);

        writer.Write((uint)7);
        writer.Write((uint)typedSection.Length);
        writer.Write(typedSection);
        writer.Write(strings);
        return stream.ToArray();
    }

    private static byte[] BuildRawSwizzleWriteTypedSection(string components)
    {
        const int headerSize = 104;

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(new byte[headerSize]);

        var functionOffset = (uint)stream.Position;
        writer.Write((byte)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write((uint)0);

        var typeOffset = (uint)stream.Position;
        writer.Write((byte)7);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)1);
        writer.Write((uint)3);
        writer.Write((uint)32);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)1);
        writer.Write((uint)1);
        writer.Write((uint)32);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)2);
        writer.Write((uint)2);
        writer.Write((uint)2);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((byte)2);
        writer.Write((uint)2);
        writer.Write((uint)4);
        writer.Write((uint)0);
        writer.Write((uint)0);

        var structOffset = (uint)stream.Position;
        var structFieldOffset = (uint)stream.Position;

        var statementOffset = (uint)stream.Position;
        writer.Write((byte)1);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write((uint)1);

        writer.Write((byte)3);
        writer.Write((uint)0);
        writer.Write((uint)3);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        var expressionOffset = (uint)stream.Position;
        writer.Write((byte)1);
        writer.Write((uint)1);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)1);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)5);
        writer.Write((uint)4);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)2);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)1);
        writer.Write((uint)2);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)3);
        writer.Write((uint)0);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);

        writer.Write((byte)12);
        writer.Write((uint)3);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)2);

        var lvalueOffset = (uint)stream.Position;
        writer.Write((byte)5);
        writer.Write((uint)3);
        writer.Write((uint)1);
        writer.Write(uint.MaxValue);
        writer.Write(uint.MaxValue);
        writer.Write((uint)4);

        var childOffset = (uint)stream.Position;
        writer.Write((uint)1);

        var argumentOffset = (uint)stream.Position;
        writer.Write((uint)2);
        writer.Write((uint)2);

        var parameterOffset = (uint)stream.Position;
        var stringOffset = (uint)stream.Position;
        var strings = BuildStringTable(["Entry", "0", "output", "1.0", components]);
        writer.Write(strings);

        var payload = stream.ToArray();
        using var headerStream = new MemoryStream(payload);
        using var header = new BinaryWriter(headerStream, System.Text.Encoding.UTF8, leaveOpen: true);
        header.Write("FTIR"u8);
        header.Write((ushort)1);
        header.Write((ushort)0);
        header.Write((byte)1);
        header.Write((byte)0);
        header.Write((ushort)headerSize);
        header.Write((uint)0);
        WriteRange(header, functionOffset, 1);
        WriteRange(header, typeOffset, 5);
        WriteRange(header, structOffset, 0);
        WriteRange(header, structFieldOffset, 0);
        WriteRange(header, statementOffset, 2);
        WriteRange(header, expressionOffset, 4);
        WriteRange(header, lvalueOffset, 1);
        WriteRange(header, childOffset, 1);
        WriteRange(header, argumentOffset, 2);
        WriteRange(header, parameterOffset, 0);
        header.Write(stringOffset);
        header.Write(checked((uint)(payload.Length - stringOffset)));
        return payload;

        static void WriteRange(BinaryWriter writer, uint offset, uint count)
        {
            writer.Write(offset);
            writer.Write(count);
        }
    }

    private static byte[] BuildStringTable(IReadOnlyList<string> values)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write((uint)values.Count);
        foreach (var value in values)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(value);
            writer.Write((uint)bytes.Length);
            writer.Write(bytes);
        }

        return stream.ToArray();
    }
}

public readonly struct RawSwizzleWriteKernel(ReadWriteBuffer<float4> output) : IGeneratedKernel<RawSwizzleWriteKernel>
{
    private readonly ReadWriteBuffer<float4> _output = output;

    public static ReadOnlySpan<byte> IR => GeneratedComputeDispatchTests.BuildRawSwizzleWriteIrForTesting("XY");

    public static KernelDescriptor Descriptor { get; } = new(
        KernelDimension.One,
        new int3(1, 1, 1),
        [new ResourceDescriptor(0, ResourceKind.Buffer, ResourceAccess.ReadWrite, typeof(float4), "output")],
        [],
        BoundsCheck: true,
        AutoDiff: false,
        nameof(RawSwizzleWriteKernel));

    public static void Bind(in RawSwizzleWriteKernel kernel, GpuKernelCommand command)
    {
        command.BindBuffer(0, ((IGpuBufferBinding)kernel._output).NativeBufferHandle);
    }
}

public readonly struct RawDuplicateSwizzleWriteKernel : IGeneratedKernel<RawDuplicateSwizzleWriteKernel>
{
    public static ReadOnlySpan<byte> IR => GeneratedComputeDispatchTests.BuildRawSwizzleWriteIrForTesting("XX");

    public static KernelDescriptor Descriptor { get; } = new(
        KernelDimension.One,
        new int3(1, 1, 1),
        [new ResourceDescriptor(0, ResourceKind.Buffer, ResourceAccess.ReadWrite, typeof(float4), "output")],
        [],
        BoundsCheck: true,
        AutoDiff: false,
        nameof(RawDuplicateSwizzleWriteKernel));

    public static void Bind(in RawDuplicateSwizzleWriteKernel kernel, GpuKernelCommand command)
    {
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct CopyKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Copy2DKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel2D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Copy3DKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel3D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct EntryAttributedComputeKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    [Entry]
    public void Run()
    {
        int i = ThreadIds.X;
        output[i] = input[i] + 3.0f;
    }

    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = -1000.0f;
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct EntryVarLocalsKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output,
    Uniform<float> scale) : IKernel1D
{
    [Entry]
    public void Apply()
    {
        var i = ThreadIds.X;
        var value = input[i];
        var offset = (float)i;
        output[i] = (value + offset) * scale.Value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct CopyWithUniformKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output, Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i];
    }
}

[Kernel]
[ThreadGroupSize(8, 1, 1)]
public readonly partial struct BoundsCheckedWriteKernel(ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = i;
    }
}

[Kernel(BoundsCheck = false)]
[ThreadGroupSize(8, 1, 1)]
public readonly partial struct UncheckedBoundsWriteKernel(ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = i;
    }
}

[Kernel]
[ThreadGroupSize(4, 4, 1)]
public readonly partial struct BoundsChecked2DWriteKernel(ReadWriteBuffer<int> output) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        output[(p.Y * 4) + p.X] = (p.Y * 10) + p.X;
    }
}

[Kernel]
[ThreadGroupSize(4, 4, 2)]
public readonly partial struct BoundsChecked3DWriteKernel(ReadWriteBuffer<int> output) : IKernel3D
{
    public void Execute()
    {
        int3 p = ThreadIds.XYZ;
        output[(p.Z * 16) + (p.Y * 4) + p.X] = (p.Z * 100) + (p.Y * 10) + p.X;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct UniformExpressionKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output, Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] * scale.Value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Float2ExpressionKernel(
    ReadOnlyBuffer<float2> input,
    ReadOnlyBuffer<float2> bias,
    ReadWriteBuffer<float2> output,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = (input[i] * scale.Value) + bias[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Float2UniformExpressionKernel(
    ReadOnlyBuffer<float2> input,
    ReadWriteBuffer<float2> output,
    Uniform<float2> offset) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] + offset.Value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Float3ExpressionKernel(
    ReadOnlyBuffer<float3> input,
    ReadWriteBuffer<float3> output,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] * scale.Value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Float4UniformExpressionKernel(
    ReadOnlyBuffer<float4> input,
    ReadWriteBuffer<float4> output,
    Uniform<float4> offset) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] + offset.Value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DotExpressionKernel(
    ReadOnlyBuffer<float3> left,
    ReadOnlyBuffer<float3> right,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ShaderMath.Dot(left[i], right[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct CrossExpressionKernel(
    ReadOnlyBuffer<float3> left,
    ReadOnlyBuffer<float3> right,
    ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ShaderMath.Cross(left[i], right[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct MultiplyFloatByLiteralKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] * 2.0f;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct UnsignedIntegerKernel(ReadOnlyBuffer<uint> input, ReadWriteBuffer<uint> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        uint value = input[i] + 3u;
        output[i] = value + input[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct UnsignedUniformKernel(
    ReadOnlyBuffer<uint> input,
    ReadWriteBuffer<uint> output,
    Uniform<uint> offset) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] + offset.Value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct AddIntegerBuffersKernel(ReadOnlyBuffer<int> left, ReadOnlyBuffer<int> right, ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = left[i] + right[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct NestedExpressionKernel(ReadOnlyBuffer<float> input, ReadOnlyBuffer<float> bias, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = (input[i] * 2.0f) + bias[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct VectorFusedMultiplyAddKernel(
    ReadOnlyBuffer<float3> left,
    ReadOnlyBuffer<float3> right,
    ReadOnlyBuffer<float3> bias,
    ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = (left[i] * right[i]) + bias[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct IntegerMultiplyAddKernel(
    ReadOnlyBuffer<int> left,
    ReadOnlyBuffer<int> right,
    ReadOnlyBuffer<int> bias,
    ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = (left[i] * right[i]) + bias[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct IntrinsicExpressionKernel(ReadOnlyBuffer<float> input, ReadOnlyBuffer<float> bias, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ShaderMath.Sqrt(input[i]) + bias[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct AdditionalScalarIntrinsicKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float value = input[i];
        output[i] = ShaderMath.InverseSqrt(value) + ShaderMath.Fract(value) + ShaderMath.Saturate(value - 24.0f);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct NormalizeIntrinsicKernel(ReadOnlyBuffer<float3> input, ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ShaderMath.Normalize(input[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct LengthIntrinsicKernel(ReadOnlyBuffer<float3> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ShaderMath.Length(input[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct VectorMathIntrinsicOverloadKernel(ReadOnlyBuffer<float3> left, ReadOnlyBuffer<float3> right, ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 a = ShaderMath.Abs(left[i]);
        float3 b = ShaderMath.Min(a, right[i]);
        float3 c = ShaderMath.Clamp(b, 0.0f, 1.0f);
        output[i] = ShaderMath.Mix(c, right[i], 0.5f);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct MultiAssignmentKernel(
    ReadOnlyBuffer<float> input,
    ReadOnlyBuffer<float> bias,
    ReadWriteBuffer<float> expressionOutput,
    ReadWriteBuffer<float> copyOutput) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        expressionOutput[i] = (input[i] * 2.0f) + bias[i];
        copyOutput[i] = input[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureCopyKernel(ReadOnlyTexture2D<Rgba32> input, ReadWriteTexture2D<Rgba32> output) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        output[p] = input[p];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureR8LoadKernel(ReadOnlyTexture2D<float4> input, ReadWriteBuffer<float> output) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        int i = (p.Y * 2) + p.X;
        output[i] = input[p].R;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureRg8LoadKernel(ReadOnlyTexture2D<float4> input, ReadWriteBuffer<float2> output) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        int i = (p.Y * 2) + p.X;
        output[i] = input[p].XY;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureR32FloatLoadKernel(ReadOnlyTexture2D<float4> input, ReadWriteBuffer<float> output) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        int i = (p.Y * 2) + p.X;
        output[i] = input[p].R;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureFloat4CopyKernel(ReadOnlyTexture2D<float4> input, ReadWriteTexture2D<float4> output) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        output[p] = input[p];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Texture3DReadKernel(ReadOnlyTexture3D<float4> input, ReadWriteBuffer<float4> output) : IKernel3D
{
    public void Execute()
    {
        int3 p = ThreadIds.XYZ;
        int i = (p.Z * 4) + (p.Y * 2) + p.X;
        output[i] = input[p];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Texture3DWriteKernel(ReadWriteTexture3D<float4> output) : IKernel3D
{
    public void Execute()
    {
        int3 p = ThreadIds.XYZ;
        float v = (p.Z * 100.0f) + (p.Y * 10.0f) + p.X;
        output[p] = new float4(p.X, p.Y, v, 1.0f);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Texture3DCopyKernel(ReadOnlyTexture3D<float4> input, ReadWriteTexture3D<float4> output) : IKernel3D
{
    public void Execute()
    {
        int3 p = new int3(ThreadIds.X, ThreadIds.Y, ThreadIds.Z);
        output[p] = input[p];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct AtomicOpsKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        _ = GpuAtomic.Add(ref output[0], input[i]);
        _ = GpuAtomic.Sub(ref output[1], 1);
        _ = GpuAtomic.Min(ref output[2], input[i]);
        _ = GpuAtomic.Max(ref output[3], input[i]);
        _ = GpuAtomic.And(ref output[4], 0xFF);
        _ = GpuAtomic.Or(ref output[5], input[i]);
        _ = GpuAtomic.Xor(ref output[6], input[i]);
        _ = GpuAtomic.Exchange(ref output[7], input[i]);
        _ = GpuAtomic.CompareExchange(ref output[8], i, input[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct AtomicAddKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        _ = GpuAtomic.Add(ref output[0], input[i]);
    }
}

[Kernel]
[ThreadGroupSize(4, 1, 1)]
public readonly partial struct SharedMemoryAtomicAddKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int local_id = LocalIds.X;
        int slot = local_id + 2;
        var shared_ints = new SharedMemory<int>(8);
        shared_ints[slot] = 0;
        GpuBarrier.Workgroup();
        _ = GpuAtomic.Add(ref shared_ints[slot], input[local_id]);
        GpuBarrier.Workgroup();
        output[local_id] = shared_ints[slot];
    }
}

[GpuStruct]
public readonly partial record struct Rg8(byte R, byte G);

[GpuStruct]
public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

[GpuStruct]
public partial struct PackedScene
{
    public float3 LightDir;
    public float Intensity;
}

[GpuStruct]
public partial struct TypedScene
{
    public float3 LightDir;
    public float Intensity;
}

[GpuStruct]
public partial struct MutableScene
{
    public float3 LightDir;
    public float Intensity;
}

[GpuStruct]
public partial struct NestedScene
{
    public TypedScene Scene;
    public float4 Weight;
}

[GpuStruct]
public readonly partial record struct CallableHitInfo(float Closest, float3 Normal, float3 Color);

[GpuStruct]
public readonly partial record struct ScaleBias(float Scale, float Bias)
{
    [Callable]
    public float Apply(float value)
    {
        return (value * Scale) + Bias;
    }
}

[GpuStruct]
public readonly partial record struct CallableGain(float Multiplier)
{
    [Callable]
    public float Apply(float value)
    {
        return value * Multiplier;
    }
}

[GpuStruct]
public partial struct MutableCounterNested
{
    public float Inner;

    [Callable]
    public void Accumulate(float amount)
    {
        Inner += amount;
    }
}

[GpuStruct]
public partial struct MutableCounter
{
    public float Value;
    public int Hits;
    public MutableCounterNested Nested;

    [Callable]
    public void Advance(float delta, int lane)
    {
        Value += delta;
        Hits++;
        Nested.Accumulate((delta * 2.0f) + lane);
    }

    [Callable]
    public void AddLocalBias(float amount)
    {
        this.Value += amount;
    }
}

public interface IMonoShape
{
    float Sdf(float3 p);
}

[GpuStruct]
public readonly partial record struct MonoSphere(float Radius) : IMonoShape
{
    [Callable]
    public float Sdf(float3 p)
    {
        return ShaderMath.Length(p) - Radius;
    }
}

[GpuStruct]
public readonly partial record struct MonoPlane(float Offset) : IMonoShape
{
    [Callable]
    public float Sdf(float3 p)
    {
        return p.Y + Offset;
    }
}

[ShaderLibrary]
public static class MonoShapeOps
{
    [Callable]
    public static float Eval<TShape>(TShape shape, float3 p)
        where TShape : IMonoShape
    {
        return shape.Sdf(p);
    }
}

public interface IMonoMutableShape
{
    void Offset(float delta);
    float Measure();
}

[GpuStruct]
public partial struct MonoMutableOffset : IMonoMutableShape
{
    public float Current;
    public float Scale;

    [Callable]
    public void Offset(float delta)
    {
        Current += delta * Scale;
    }

    [Callable]
    public float Measure()
    {
        return Current;
    }
}

[ShaderLibrary]
public static class MonoMutableOps
{
    [Callable]
    public static float EvalAfterOffset<TShape>(TShape shape, float delta)
        where TShape : IMonoMutableShape
    {
        shape.Offset(delta);
        return shape.Measure();
    }
}

[GpuStruct]
public readonly partial record struct ComplexScaleBias(float Scale, float Bias, int Mode)
{
    [Callable]
    public float Apply(CallableGain gain, float value)
    {
        float shaped = this.Shape(value);
        float adjusted = gain.Apply(shaped);
        return Mode == 0 ? adjusted : adjusted + Bias;
    }

    [Callable]
    public float Shape(float value)
    {
        return this.ScaleOnly(value) + Bias;
    }

    [Callable]
    public float ScaleOnly(float value)
    {
        return value * Scale;
    }

    [Callable]
    public float ScaleOnly(int value)
    {
        return value * 100.0f;
    }

    [Callable]
    public float Unused(float value)
    {
        return value - 1000.0f;
    }
}

[GpuStruct]
public partial struct MatrixScene
{
    public float Weight;
    public float2x2 Transform;
    public float Bias;
}

[GpuStruct]
public partial struct BoolVectorScene
{
    public bool Enabled;
    public bool3 Mask;
    public float Weight;
}

[GpuStruct]
public partial struct InnerArrayScene
{
    public float Value;
}

[GpuStruct]
public partial struct ArrayScene
{
    public GpuArray4<float3> Directions;
    public float Weight;
}

[GpuStruct]
public partial struct NestedArrayScene
{
    public GpuArray3<InnerArrayScene> Items;
    public float Weight;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct StructFieldReadKernel(
    ReadOnlyBuffer<TypedScene> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        TypedScene scene = input[i];
        output[i] = scene.Intensity + scene.LightDir.X + scene.LightDir.Y + scene.LightDir.Z;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct StructFieldWriteKernel(
    ReadOnlyBuffer<float> intensity,
    ReadWriteBuffer<MutableScene> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i].LightDir = new float3(i + 1, i + 2, i + 3);
        output[i].Intensity = intensity[i] + i;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct NestedStructFieldReadKernel(
    ReadOnlyBuffer<NestedScene> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        NestedScene scene = input[i];
        output[i] = scene.Scene.Intensity
            + scene.Scene.LightDir.X
            + scene.Scene.LightDir.Y
            + scene.Scene.LightDir.Z
            + scene.Weight.X;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct NestedStructFieldWriteKernel(
    ReadOnlyBuffer<float> values,
    ReadWriteBuffer<NestedScene> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float value = values[i] + i;
        output[i].Scene.LightDir = new float3(value, value + 1, value + 2);
        output[i].Scene.Intensity = value + 5;
        output[i].Weight = new float4(value, value * 0.5f, value * 0.2f, 1);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct CallableStructReturnKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        CallableHitInfo hit = MakeHit(input[i], i);
        output[i] = hit.Closest + hit.Normal.X + hit.Normal.Y + hit.Normal.Z + hit.Color.X;
    }

    [Callable]
    private static CallableHitInfo MakeHit(float value, int i)
    {
        float3 normal = new float3(i + 1, i + 2, i + 3);
        if (i == 0)
        {
            normal = new float3(1, 2, 3);
        }

        return new CallableHitInfo(value * 2, normal, new float3(value + 7, 0, 0));
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GpuStructInstanceCallableKernel(
    ReadOnlyBuffer<ScaleBias> parameters,
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = parameters[i].Apply(input[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GpuStructMutatingInstanceCallableKernel(
    ReadWriteBuffer<MutableCounter> counters,
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        counters[i].Advance(input[i], i);

        MutableCounter local = counters[i];
        local.AddLocalBias(1.5f);
        counters[i] = local;

        output[i] = counters[i].Value + counters[i].Nested.Inner + counters[i].Hits;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GenericInterfaceShapeKernel(
    ReadOnlyBuffer<MonoSphere> spheres,
    ReadOnlyBuffer<MonoPlane> planes,
    ReadOnlyBuffer<float3> points,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = MonoShapeOps.Eval(spheres[i], points[i])
            + MonoShapeOps.Eval(planes[i], points[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GenericMutatingInterfaceShapeKernel(
    ReadOnlyBuffer<MonoMutableOffset> shapes,
    ReadOnlyBuffer<float> deltas,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = MonoMutableOps.EvalAfterOffset(shapes[i], deltas[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct ComplexGpuStructInstanceCallableKernel(
    ReadOnlyBuffer<ComplexScaleBias> curves,
    ReadOnlyBuffer<CallableGain> gains,
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        ComplexScaleBias curve = curves[i];
        CallableGain gain = gains[i];
        output[i] = curve.Apply(gain, input[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct StructArrayReadKernel(
    ReadOnlyBuffer<ArrayScene> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        int slot = i + 1;
        ArrayScene scene = input[i];
        float3 direction = scene.Directions[slot];
        output[i] = direction.X + direction.Y + direction.Z + scene.Weight;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct StructArrayWriteKernel(
    ReadOnlyBuffer<float> values,
    ReadWriteBuffer<NestedArrayScene> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float value = values[i] + i;
        output[i].Items[0].Value = value;
        output[i].Items[1].Value = value + 1;
        output[i].Items[2].Value = value + 2;
        output[i].Weight = value * 2;
    }
}

public class ControlFlowDispatchTests
{
    [Fact]
    public void ShaderInspectionLowersIfConditionThroughEasyGpuBuilder()
    {
        var glsl = ShaderInspection.GetGLSL<IfThresholdKernel>();

        Assert.Contains("#version", glsl, StringComparison.Ordinal);
        Assert.Contains("if ", glsl, StringComparison.Ordinal);
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("_i < 256", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ShaderInspectionLowersForLoopThroughEasyGpuBuilder()
    {
        var glsl = ShaderInspection.GetGLSL<ForLoopSumKernel>();

        Assert.Contains("#version", glsl, StringComparison.Ordinal);
        Assert.Contains("for ", glsl, StringComparison.Ordinal);
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("_i < 256", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ShaderInspectionLowersWhileLoopThroughEasyGpuBuilder()
    {
        var glsl = ShaderInspection.GetGLSL<WhileAccumulateKernel>();

        Assert.Contains("#version", glsl, StringComparison.Ordinal);
        Assert.Contains("while ", glsl, StringComparison.Ordinal);
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void DispatchExecutesIfThresholdThroughEasyGpu()
    {
        using var input = GPU.CreateBuffer<float>([-2, -1, 0, 1, 2, 3]);
        using var output = GPU.CreateBuffer<float>(6);
        var kernel = new IfThresholdKernel(input.AsReadOnly(), output.AsReadWrite());

        GPU.Dispatch(kernel, 6);

        // Condition is input[i] > 0, properly lowered through expression tree
        Assert.Equal([0, 0, 0, 1, 2, 3], output.ToArray());
    }
}

public class NeuralNetworkDispatchTests
{
    [Fact]
    public void ShaderInspectionLowersDotProductThroughEasyGpuWithUniform()
    {
        var glsl = ShaderInspection.GetGLSL<DotProductKernel>();

        Assert.Contains("#version", glsl, StringComparison.Ordinal);
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct IfThresholdKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        if (input[i] > 0)
            output[i] = input[i];
        else
            output[i] = 0;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DynamicForSumKernel(ReadOnlyBuffer<int> counts, ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float sum = 0;
        for (int j = 0; j < counts[i]; j++)
        {
            sum += input[i + j];
        }

        output[i] = sum;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct ForLoopSumKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float sum = 0;
        for (int k = 0; k < 4; k++)
            sum += input[k];
        output[i] = sum;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct WhileAccumulateKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float sum = 0;
        int k = 0;
        while (k < 4)
        {
            sum += input[k];
            k++;
        }
        output[i] = sum;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DoWhileAccumulateKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float sum = 0;
        int k = 0;
        do
        {
            sum += input[i + k];
            k++;
        }
        while (k < 3);

        output[i] = sum;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BreakContinueLoopKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float sum = 0;
        for (int k = 0; k < 5; k++)
        {
            if (k == 1)
            {
                continue;
            }

            if (k == 3)
            {
                break;
            }

            sum += input[k];
        }

        output[i] = sum;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct ConditionlessForBreakKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        for (;;)
        {
            output[i] = input[i] * 2.0f;
            break;
        }
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct LogicalPredicateKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        bool keep = (input[i] > 1.0f || input[i] < -1.0f) && !(input[i] == -1.0f);
        output[i] = keep ? input[i] : 0.0f;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct UnaryModuloKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = -(input[i] + (i % 2));
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BitwiseShiftKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        int mixed = (input[i] | 1) ^ (i & 1);
        output[i] = mixed << 1;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct VectorConstructorKernel(ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = new float3(i, i + 1, i + 2);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct SwizzleReadKernel(ReadOnlyBuffer<float4> input, ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float4 value = input[i];
        float2 xy = value.XY;
        output[i] = new float3(xy, value.Z + value.A);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DirectColorSwizzleKernel(ReadOnlyBuffer<float4> input, ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i].RGB;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct ExpandedSwizzleKernel(ReadOnlyBuffer<float4> input, ReadWriteBuffer<float4> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float4 value = input[i];
        float2 yx = value.YX;
        float3 zxy = value.ZXY;
        float4 wzyx = value.WZYX;
        float4 bgra = value.BGRA;
        output[i] = new float4(yx.X + zxy.X, yx.Y + zxy.Y, wzyx.X + bgra.X, wzyx.W + bgra.W);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureCoordinateSwizzleKernel(ReadOnlyBuffer<float4> input, ReadWriteBuffer<float4> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float4 value = input[i];
        float2 ts = value.TS;
        float3 pst = value.PST;
        float4 qpts = value.QPTS;
        output[i] = new float4(ts.X, ts.Y, pst.X, qpts.W);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct IntegerVectorConstructorKernel(ReadWriteBuffer<int3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = new int3(i, i + 10, i * 2);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BoolVectorConstructorKernel(ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        bool2 pair = new bool2(i == 0, i != 1);
        bool3 triple = new bool3(pair.X, pair.Y, i < 3);
        bool4 quad = new bool4(triple.X, triple.Y, triple.Z, !triple.X);
        int value = quad.X ? 1 : 0;
        value += quad.Y ? 2 : 0;
        value += quad.Z ? 4 : 0;
        value += quad.W ? 8 : 0;
        output[i] = value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct NumericCastKernel(
    ReadOnlyBuffer<int> ints,
    ReadOnlyBuffer<float> floats,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float fromInt = ints[i];
        int truncated = (int)floats[i];
        output[i] = fromInt + truncated;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct VectorCastKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 widened = new float3(input[i], input[i] + 1, input[i] + 2);
        output[i] = new int3((int)widened.X, (int)widened.Y, (int)widened.Z);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Matrix2VectorMultiplyKernel(ReadWriteBuffer<float2> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2x2 transform = new float2x2(new float2(1, 2), new float2(3, 4));
        output[i] = transform * new float2(i, i + 1);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Matrix3MultiplyKernel(ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3x3 left = new float3x3(
            new float3(1, 0, 0),
            new float3(0, 1, 0),
            new float3(0, 0, 1));
        float3x3 right = new float3x3(
            new float3(i + 1, 0, 0),
            new float3(0, i + 2, 0),
            new float3(0, 0, i + 3));
        float3x3 combined = left * right;
        output[i] = combined.C1;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Matrix4VectorMultiplyKernel(ReadWriteBuffer<float4> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float4x4 transform = new float4x4(
            new float4(1, 0, 0, 0),
            new float4(0, 1, 0, 0),
            new float4(0, 0, 1, 0),
            new float4(i, i + 1, i + 2, 1));
        output[i] = transform * new float4(1, 2, 3, 1);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Matrix3UniformMultiplyKernel(
    ReadWriteBuffer<float3> output,
    Uniform<float3x3> transform,
    Uniform<float3x3> offsetTransform,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 transformed = transform.Value * new float3(i, i + 1, 1);
        float3 offset = offsetTransform.Value * new float3(1, 2, 1);
        output[i] = transformed + (offset * scale.Value);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct SquareMatrixUniformKernel(
    ReadWriteBuffer<float4> output,
    Uniform<float2x2> matrix2,
    Uniform<float3x3> matrix3,
    Uniform<float4x4> matrix4,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 from2 = matrix2.Value * new float2(i + 1, 2);
        float3 from3 = matrix3.Value * new float3(1, i + 1, 1);
        float4 from4 = matrix4.Value * new float4(1, 2, 3, 1);
        output[i] = new float4(from2.X + from3.X, from2.Y + from3.Y, from4.Z, from4.W * scale.Value);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct MatrixColumnAccessKernel(ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float4x4 transform = new float4x4(
            new float4(1, 0, 0, 0),
            new float4(0, 1, 0, 0),
            new float4(i, i + 1, i + 2, 0),
            new float4(0, 0, 0, 1));
        output[i] = transform.C2.XYZ;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct MatrixMathIntrinsicKernel(ReadWriteBuffer<float4> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float4x4 transform = new float4x4(
            new float4(1, 0, 0, 0),
            new float4(0, 2, 0, 0),
            new float4(0, 0, 4, 0),
            new float4(0, 0, 0, 1));
        float4x4 transposed = ShaderMath.Transpose(transform);
        float4x4 inverse = Hlsl.Inverse(transform);
        float4x4 componentProduct = ShaderMath.Hadamard(transposed, inverse);
        float determinant = ShaderMath.Determinant(transform);
        float4 scaled = Hlsl.Mul(componentProduct, new float4(determinant, determinant, determinant, 1));
        output[i] = ShaderMath.Mul(transform, scaled) * (i + 1);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct ThreadIdsXyLinearIndexKernel(ReadWriteBuffer<int> output) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        int index = (p.Y * 3) + p.X;
        output[index] = (p.Y * 10) + p.X;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct ThreadIdsXyzLinearIndexKernel(ReadWriteBuffer<int> output) : IKernel3D
{
    public void Execute()
    {
        int3 p = ThreadIds.XYZ;
        int index = (p.Z * 4) + (p.Y * 2) + p.X;
        output[index] = (p.Z * 100) + (p.Y * 10) + p.X;
    }
}

[Kernel]
[ThreadGroupSize(4, 1, 1)]
public readonly partial struct LocalIdsKernel(ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int local_id = LocalIds.X;
        output[ThreadIds.X] = local_id;
    }
}

[Kernel]
[ThreadGroupSize(1)]
public readonly partial struct ReservedLocalIdentifierKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> destination) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float output = input[i];
        float texture = output + 1;
        destination[i] = texture;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TypedCallableKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Twice(input[i]);
    }

    [Callable]
    private static float Twice(float value)
    {
        return value * 2;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TypedCallableControlFlowKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Shape(input[i]);
    }

    [Callable]
    private static float Shape(float value)
    {
        float scaled = value * 3;
        if (scaled > 20)
        {
            return scaled - 18;
        }

        if (scaled > 0)
        {
            return scaled;
        }

        return value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TypedCallableParameterReassignmentKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Shape(input[i]);
    }

    [Callable]
    private static float Shape(float value)
    {
        value += 2;
        if (value > 3)
        {
            value = (value * 2) + 1;
        }

        return value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TypedCallableOverloadKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Shape(input[i]);
    }

    [Callable]
    private static float Shape(float value)
    {
        return value * 2;
    }

    [Callable]
    private static int Shape(int value)
    {
        return value + 10;
    }
}

[Kernel]
[ThreadGroupSize(4, 1, 1)]
public readonly partial struct SharedFloatCopyKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int local_id = LocalIds.X;
        var shared_values = new SharedMemory<float>(4);
        shared_values[local_id] = input[local_id];
        GpuBarrier.Workgroup();
        output[local_id] = shared_values[3 - local_id];
    }
}

[Kernel]
[ThreadGroupSize(4, 1, 1)]
public readonly partial struct SharedIntDynamicIndexKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int local_id = LocalIds.X;
        int slot = local_id + 2;
        var shared_ints = new SharedMemory<int>(8);
        shared_ints[slot] = input[local_id];
        GpuBarrier.Workgroup();
        output[local_id] = shared_ints[slot];
    }
}

[Kernel]
[ThreadGroupSize(4, 1, 1)]
public readonly partial struct MultipleSharedMemoryKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int local_id = LocalIds.X;
        var shared_left = new SharedMemory<float>(4);
        var shared_right = new SharedMemory<int>(4);
        shared_left[local_id] = input[local_id];
        shared_right[local_id] = local_id;
        GpuBarrier.Workgroup();
        output[local_id] = shared_left[local_id] + shared_right[local_id];
    }
}

[Kernel]
[ThreadGroupSize(2, 1, 1)]
public readonly partial struct SharedVectorMemoryKernel(ReadWriteBuffer<float2> output) : IKernel1D
{
    public void Execute()
    {
        int local_id = LocalIds.X;
        var shared_vectors = new SharedMemory<float2>(2);
        shared_vectors[local_id] = new float2(local_id, local_id + 1);
        GpuBarrier.Workgroup();
        output[local_id] = shared_vectors[local_id];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TypedBarrierKindsKernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        GpuBarrier.Workgroup();
        GpuBarrier.Memory();
        GpuBarrier.Full();
        output[i] = 1;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DotProductKernel(
    ReadOnlyBuffer<float> left,
    ReadOnlyBuffer<float> right,
    ReadWriteBuffer<float> output,
    Uniform<int> count) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = left[i] * right[i] * count.Value;
    }
}
