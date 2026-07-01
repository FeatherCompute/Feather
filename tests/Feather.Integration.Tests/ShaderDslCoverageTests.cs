using Feather.Interop;
using Feather.Math;
using Feather.Resources;
using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace Feather.Integration.Tests;

/// <summary>
/// DSL coverage smoke tests. Inspection assertions check typed lowering and
/// dispatch assertions check current runtime behavior; typed-only tests in
/// GeneratedComputeDispatchTests prove the canonical Roslyn typed-IR path.
/// </summary>
public class ShaderDslCoverageTests
{
    // ── Baseline: buffer copy (must work) ──────────────────────────────

    [Fact]
    public void BufferCopy_GeneratesValidGlslAndDispatches()
    {
        var glsl = ShaderInspection.GetGLSL<DslCopyKernel>();
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new DslCopyKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        Assert.Equal([1, 2, 3, 4], output.ToArray());
    }

    // ── Local variables ────────────────────────────────────────────────

    [Fact]
    public void LocalVariable_DeclaresThreadIdAsInt_GeneratesValidGlsl()
    {
        var glsl = ShaderInspection.GetGLSL<DslLocalVarKernel>();

        // The GLSL must declare a local int variable initialized from thread ID
        // and use it as buffer index. The thread ID reference comes through the
        // local variable declaration: int i = int(gl_GlobalInvocationID.x);
        bool hasThreadId = glsl.Contains("gl_GlobalInvocationID.x", StringComparison.Ordinal);
        bool hasLocalVar = glsl.Contains("int i", StringComparison.Ordinal);
        Assert.True(hasThreadId || hasLocalVar,
            "GLSL must reference the thread ID either directly or via local variable declaration");
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void LocalVariable_CopyThroughLocal_Dispatches()
    {
        using var input = GPU.CreateBuffer<float>([10, 20, 30, 40]);
        using var output = GPU.CreateBuffer<float>(4);
        // DslLocalVarKernel uses int i = ThreadIds.X then copies output[i] = input[i]
        GPU.Dispatch(new DslLocalVarKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        Assert.Equal([10, 20, 30, 40], output.ToArray());
    }

    // ── If/else control flow ───────────────────────────────────────────

    [Fact]
    public void IfElse_GeneratesStructuredIfNotRawSourceInjection()
    {
        var glsl = ShaderInspection.GetGLSL<DslIfElseKernel>();

        Assert.Contains("if ", glsl, StringComparison.Ordinal);
        Assert.Contains("else", glsl, StringComparison.Ordinal);
        // Must NOT contain the hardcoded C# source text (e.g. "i < 256" as raw string)
        Assert.DoesNotContain("_i < 256", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void IfElse_ConditionWiredFromExpressionTree_CorrectThresholdDispatch()
    {
        var glsl = ShaderInspection.GetGLSL<DslIfElseKernel>();
        Assert.Contains("if (", glsl, StringComparison.Ordinal);
        Assert.Contains("else", glsl, StringComparison.Ordinal);
        // Condition must NOT be hardcoded "true"
        Assert.DoesNotContain("if (true)", glsl, StringComparison.Ordinal);

        using var input = GPU.CreateBuffer<float>([-2, -1, 0, 1, 2, 3]);
        using var output = GPU.CreateBuffer<float>(6);
        GPU.Dispatch(new DslIfElseKernel(input.AsReadOnly(), output.AsReadWrite()), 6);
        // Condition is input[i] > 0 — threshold at zero
        Assert.Equal([0, 0, 0, 1, 2, 3], output.ToArray());
    }

    // ── Compound assignments ──────────────────────────────────────────

    [Fact]
    public void CompoundAssignment_AddAssign_Dispatches()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>([10, 20, 30, 40]);
        GPU.Dispatch(new DslCompoundAddKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        Assert.Equal([11, 22, 33, 44], output.ToArray());
    }

    // ── Intrinsic calls ────────────────────────────────────────────────

    [Fact]
    public void ShaderMath_GeneratesIntrinsicCalls()
    {
        var glsl = ShaderInspection.GetGLSL<DslIntrinsicKernel>();

        Assert.Contains("sin", glsl, StringComparison.Ordinal);
        Assert.Contains("sqrt", glsl, StringComparison.Ordinal);
        Assert.Contains("gl_GlobalInvocationID.x", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ShaderMath_Dispatch()
    {
        using var input = GPU.CreateBuffer<float>([0, 1, 4, 9]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new DslIntrinsicKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        // input = [0, 1, 4, 9]; output[i] = sin(input[i]) + sqrt(input[i])
        Assert.Equal([0, MathF.Sin(1) + MathF.Sqrt(1), MathF.Sin(4) + MathF.Sqrt(4), MathF.Sin(9) + MathF.Sqrt(9)], output.ToArray());
    }

    // ── For loop ───────────────────────────────────────────────────────

    [Fact]
    public void ForLoop_WithInit_GeneratesValidGlsl()
    {
        var glsl = ShaderInspection.GetGLSL<DslForLoopKernel>();
        Assert.Contains("for ", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("_i < 256", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ForLoop_WithInit_Dispatches()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new DslForLoopKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        Assert.Equal([2, 4, 6, 8], output.ToArray());
    }

    // ── Ternary ───────────────────────────────────────────────────────

    [Fact]
    public void Ternary_ConditionalExpression_Dispatches()
    {
        using var input = GPU.CreateBuffer<float>([-2, -1, 1, 2]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new DslTernaryKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        // output[i] = input[i] > 0 ? input[i] : 0
        Assert.Equal([0, 0, 1, 2], output.ToArray());
    }

    // ── Float2 vector math ─────────────────────────────────────────────

    [Fact]
    public void Float2_ExpressionDispatch()
    {
        using var a = GPU.CreateBuffer<float2>([new float2(1, 2), new float2(3, 4)]);
        using var b = GPU.CreateBuffer<float2>([new float2(10, 20), new float2(30, 40)]);
        using var output = GPU.CreateBuffer<float2>(2);
        GPU.Dispatch(new DslFloat2Kernel(a.AsReadOnly(), b.AsReadOnly(), output.AsReadWrite()), 2);
        Assert.Equal([new float2(11, 22), new float2(33, 44)], output.ToArray());
    }

}

/// <summary>
/// End-to-end coverage for the basic shader DSL features that must cooperate in
/// one real typed EasyGPU dispatch, not just as isolated syntax checks.
/// </summary>
public class BasicDslStressCoverageTests
{
    [Fact]
    public void BasicDslStressKernel_DispatchesThroughTypedEasyGpuAndMatchesCpuReference()
    {
        var records = new[]
        {
            new DslStressRecord
            {
                Inner = new DslStressInner { Direction = new float3(1.0f, 2.0f, 3.0f), Weight = 0.5f },
                Uv = new float2(0.25f, 0.75f),
                IterationLimit = 2,
                Bias = 1.0f,
                Flags = 1
            },
            new DslStressRecord
            {
                Inner = new DslStressInner { Direction = new float3(-1.0f, 0.5f, 2.0f), Weight = 2.0f },
                Uv = new float2(1.0f, -0.5f),
                IterationLimit = 4,
                Bias = -1.0f,
                Flags = 2
            },
            new DslStressRecord
            {
                Inner = new DslStressInner { Direction = new float3(0.25f, -1.5f, 0.75f), Weight = 1.25f },
                Uv = new float2(0.0f, 2.0f),
                IterationLimit = 1,
                Bias = 0.25f,
                Flags = 0
            }
        };

        AssertStressStructLayouts(records[0]);

        var glsl = ShaderInspection.GetGLSL<BasicDslStressKernel>();
        Assert.Contains("for ", glsl, StringComparison.Ordinal);
        Assert.Contains("while ", glsl, StringComparison.Ordinal);
        Assert.Contains("do", glsl, StringComparison.Ordinal);
        Assert.Contains("if ", glsl, StringComparison.Ordinal);
        Assert.Contains("continue", glsl, StringComparison.Ordinal);
        Assert.Contains("break", glsl, StringComparison.Ordinal);
        Assert.Contains("BasicDslStressKernel_Shape", glsl, StringComparison.Ordinal);
        Assert.Contains("BasicDslStressKernel_InnerCurve", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var input = GPU.CreateBuffer<DslStressRecord>(records, BufferAccess.ReadOnly);
        using var output = GPU.CreateBuffer<DslStressResult>(records.Length, BufferAccess.ReadWrite);
        using var scalarOutput = GPU.CreateBuffer<float>(records.Length);
        using var intOutput = GPU.CreateBuffer<int>(records.Length);

        const int globalIterations = 3;
        const float globalBias = 0.25f;
        var path = GPU.DispatchAndGetPath(
            new BasicDslStressKernel(
                input.AsReadOnly(),
                output.AsReadWrite(),
                scalarOutput.AsReadWrite(),
                intOutput.AsReadWrite(),
                new Uniform<int>(globalIterations),
                new Uniform<float>(globalBias)),
            records.Length);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);

        var actualResults = output.ToArray();
        var actualScores = scalarOutput.ToArray();
        var actualInts = intOutput.ToArray();
        for (var i = 0; i < records.Length; i++)
        {
            var expected = EvaluateStressReference(records[i], i, globalIterations, globalBias);
            AssertNear(expected.Color, actualResults[i].Color);
            AssertNear(expected.Normal, actualResults[i].Normal);
            AssertNear(expected.Score, actualResults[i].Score);
            Assert.Equal(expected.Steps, actualResults[i].Steps);
            AssertNear(expected.Score, actualScores[i]);
            Assert.Equal(expected.IntegerValue, actualInts[i]);
        }
    }

    private static void AssertStressStructLayouts(DslStressRecord sample)
    {
        var inner = GpuStructLayout<DslStressInner>();
        Assert.Equal(GpuLayout.Std430, inner.Layout);
        Assert.Equal(16, inner.SizeInBytes);
        Assert.Equal(16, inner.Alignment);
        AssertField(inner, nameof(DslStressInner.Direction), 0, 12, 16);
        AssertField(inner, nameof(DslStressInner.Weight), 12, 4, 4);

        var record = GpuStructLayout<DslStressRecord>();
        Assert.Equal(GpuLayout.Std430, record.Layout);
        Assert.Equal(48, record.SizeInBytes);
        Assert.Equal(16, record.Alignment);
        AssertField(record, nameof(DslStressRecord.Inner), 0, 16, 16);
        AssertField(record, nameof(DslStressRecord.Uv), 16, 8, 8);
        AssertField(record, nameof(DslStressRecord.IterationLimit), 24, 4, 4);
        AssertField(record, nameof(DslStressRecord.Bias), 28, 4, 4);
        AssertField(record, nameof(DslStressRecord.Flags), 32, 4, 4);

        var result = GpuStructLayout<DslStressResult>();
        Assert.Equal(GpuLayout.Std430, result.Layout);
        Assert.Equal(48, result.SizeInBytes);
        Assert.Equal(16, result.Alignment);
        AssertField(result, nameof(DslStressResult.Color), 0, 16, 16);
        AssertField(result, nameof(DslStressResult.Normal), 16, 12, 16);
        AssertField(result, nameof(DslStressResult.Score), 28, 4, 4);
        AssertField(result, nameof(DslStressResult.Steps), 32, 4, 4);

        var bytes = Enumerable.Repeat((byte)0xCD, record.SizeInBytes).ToArray();
        GpuStructPack<DslStressRecord>([sample], bytes);
        Assert.Equal(sample.Inner.Direction.X, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(0, 4)));
        Assert.Equal(sample.Inner.Direction.Y, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(4, 4)));
        Assert.Equal(sample.Inner.Direction.Z, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(8, 4)));
        Assert.Equal(sample.Inner.Weight, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(12, 4)));
        Assert.Equal(sample.Uv.X, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(16, 4)));
        Assert.Equal(sample.Uv.Y, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(20, 4)));
        Assert.Equal(sample.IterationLimit, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(24, 4)));
        Assert.Equal(sample.Bias, BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(28, 4)));
        Assert.Equal(sample.Flags, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(32, 4)));
        Assert.All(bytes[36..48], value => Assert.Equal(0, value));
    }

    private static void AssertField(GpuStructLayout layout, string name, int offset, int size, int alignment)
    {
        var field = layout.Fields.Single(field => field.Name == name);
        Assert.Equal(offset, field.Offset);
        Assert.Equal(size, field.SizeInBytes);
        Assert.Equal(alignment, field.Alignment);
    }

    private static GpuStructLayout GpuStructLayout<T>()
        where T : unmanaged, IGpuStruct<T>
        => T.Layout;

    private static void GpuStructPack<T>(ReadOnlySpan<T> source, Span<byte> destination)
        where T : unmanaged, IGpuStruct<T>
        => T.Pack(source, destination);

    private static DslStressExpected EvaluateStressReference(
        DslStressRecord record,
        int index,
        int globalIterations,
        float globalBias)
    {
        var transform = new float2x2(new float2(1.0f, 2.0f), new float2(3.0f, 4.0f));
        var transformed = transform * new float2(record.Uv.Y, record.Uv.X);
        var mixed = new float3(transformed.X, transformed.Y, record.Inner.Direction.Z) + record.Inner.Direction;
        var dot = ShaderMath.Dot(mixed, new float3(1.0f, 0.5f, -1.0f));
        var cross = ShaderMath.Cross(record.Inner.Direction, new float3(0.0f, 1.0f, 0.0f));
        var shaped = ShapeReference(dot + record.Inner.Weight, record.Bias + globalBias);
        var limit = record.IterationLimit + globalIterations;
        var accumulator = 0.0f;

        for (var k = 0; k < limit; k++)
        {
            if ((k % 2) == 1)
            {
                continue;
            }

            accumulator += k * 0.5f;
            if (accumulator > 3.0f)
            {
                break;
            }
        }

        var whileIndex = 0;
        while (whileIndex < record.IterationLimit)
        {
            accumulator += 0.25f;
            whileIndex++;
        }

        var doIndex = 0;
        do
        {
            accumulator += 0.125f;
            doIndex++;
        }
        while (doIndex < 2);

        var keep = (((record.Flags & 1) == 1) || shaped > 0.0f) && !(record.Flags == 0);
        var score = keep ? shaped + accumulator : -shaped;
        var clamped = ShaderMath.Clamp(score, -32.0f, 32.0f);
        var normal = ShaderMath.Normalize(cross + new float3(0.01f, 0.0f, 0.0f));
        var color = new float4(normal.XY, clamped, 1.0f);
        var integerValue = ((int)clamped) + ((record.Flags << 1) ^ (index & 1));

        return new DslStressExpected(color, normal, clamped, limit, integerValue);
    }

    private static float ShapeReference(float value, float bias)
    {
        var curved = InnerCurveReference(value, bias);
        if (curved > 4.0f)
        {
            return curved - 1.0f;
        }

        return curved + 2.0f;
    }

    private static float InnerCurveReference(float value, float bias)
        => (value * 0.75f) + bias;

    private static void AssertNear(float4 expected, float4 actual, float tolerance = 1e-4f)
    {
        AssertNear(expected.X, actual.X, tolerance);
        AssertNear(expected.Y, actual.Y, tolerance);
        AssertNear(expected.Z, actual.Z, tolerance);
        AssertNear(expected.W, actual.W, tolerance);
    }

    private static void AssertNear(float3 expected, float3 actual, float tolerance = 1e-4f)
    {
        AssertNear(expected.X, actual.X, tolerance);
        AssertNear(expected.Y, actual.Y, tolerance);
        AssertNear(expected.Z, actual.Z, tolerance);
    }

    private static void AssertNear(float expected, float actual, float tolerance = 1e-4f)
        => Assert.InRange(MathF.Abs(actual - expected), 0, tolerance);

    private readonly record struct DslStressExpected(
        float4 Color,
        float3 Normal,
        float Score,
        int Steps,
        int IntegerValue);
}

/// <summary>
/// End-to-end torture coverage for syntax combinations that are individually
/// legal but historically easy to break during typed IR lowering.
/// </summary>
public class PathologicalDslLoweringCoverageTests
{
    [Fact]
    public void PathologicalDslTortureKernel_DispatchesThroughTypedEasyGpuAndMatchesCpuReference()
    {
        const int count = 8;
        const int dynamicStride = 3;
        const float uniformBias = 0.125f;
        var source = Enumerable.Range(0, 64)
            .Select(i => ((i % 11) - 5) * 0.375f + (i * 0.03125f))
            .ToArray();
        var scalarSentinel = Enumerable.Repeat(-999.0f, count * PathologicalDslConstants.OutputStride).ToArray();

        var expected = EvaluatePathologicalReference(source, count, dynamicStride, uniformBias, scalarSentinel);

        var glsl = ShaderInspection.GetGLSL<PathologicalDslTortureKernel>();
        Assert.Contains("fe_in", glsl, StringComparison.Ordinal);
        Assert.Contains("fe_output", glsl, StringComparison.Ordinal);
        Assert.Contains("fe_texture", glsl, StringComparison.Ordinal);
        Assert.Contains("fe_sampler", glsl, StringComparison.Ordinal);
        Assert.Contains("fe_shared", glsl, StringComparison.Ordinal);
        Assert.Contains("for ", glsl, StringComparison.Ordinal);
        Assert.Contains("while ", glsl, StringComparison.Ordinal);
        Assert.Contains("do", glsl, StringComparison.Ordinal);
        Assert.Contains("continue", glsl, StringComparison.Ordinal);
        Assert.Contains("break", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var input = GPU.CreateBuffer<float>(source, BufferAccess.ReadOnly);
        using var vectorOutput = GPU.CreateBuffer<float4>(count, BufferAccess.ReadWrite);
        using var scalarOutput = GPU.CreateBuffer<float>(scalarSentinel, BufferAccess.ReadWrite);

        var path = GPU.DispatchAndGetPath(
            new PathologicalDslTortureKernel(
                input.AsReadOnly(),
                vectorOutput.AsReadWrite(),
                scalarOutput.AsReadWrite(),
                new Uniform<int>(dynamicStride),
                new Uniform<float>(uniformBias)),
            count);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);

        var actualVectors = vectorOutput.ToArray();
        var actualScalars = scalarOutput.ToArray();
        for (var i = 0; i < count; i++)
        {
            AssertNear(expected.Vectors[i], actualVectors[i], 2e-3f);
        }

        for (var i = 0; i < expected.Scalars.Length; i++)
        {
            AssertNear(expected.Scalars[i], actualScalars[i], 2e-3f);
        }
    }

    private static PathologicalExpected EvaluatePathologicalReference(
        float[] input,
        int count,
        int dynamicStride,
        float uniformBias,
        float[] scalarSentinel)
    {
        var preShared = new float[count];
        var textureValues = new float[count];
        var samplerValues = new float[count];
        var outIndices = new int[count];
        for (var i = 0; i < count; i++)
        {
            var pre = EvaluatePathologicalPreShared(input, i, dynamicStride, uniformBias);
            preShared[i] = pre.Shaped;
            textureValues[i] = pre.Texture;
            samplerValues[i] = pre.Sampler;
            outIndices[i] = pre.OutputIndex;
        }

        var vectors = new float4[count];
        var scalars = scalarSentinel.ToArray();
        for (var i = 0; i < count; i++)
        {
            var layout = i % PathologicalDslConstants.Tile;
            var groupBase = i - layout;
            var neighbor = preShared[groupBase + (PathologicalDslConstants.Tile - 1 - layout)];
            var uniform = preShared[i] + (neighbor * 0.125f);
            var xy = new float2(uniform, textureValues[i]);
            var transform = new float2x2(new float2(1.0f, 0.25f), new float2(-0.5f, 0.75f));
            var transformed = transform * xy;
            var normal = ShaderMath.Normalize(
                new float3(transformed.X, transformed.Y, samplerValues[i]) + new float3(0.01f, 0.02f, 0.03f));
            var clamped = ShaderMath.Clamp(uniform + ShaderMath.Dot(normal, new float3(0.25f, -0.5f, 0.75f)), -32.0f, 32.0f);
            var vector = new float4(normal.XY, clamped, 1.0f);

            vectors[i] = vector;
            scalars[outIndices[i]] = vector.Z + (outIndices[i] % 3);
        }

        return new PathologicalExpected(vectors, scalars);
    }

    private static PathologicalPreShared EvaluatePathologicalPreShared(
        float[] input,
        int i,
        int dynamicStride,
        float uniformBias)
    {
        var layout = i % PathologicalDslConstants.Tile;
        const int localScale = 2;
        var baseIndex = ((i * PathologicalDslConstants.InputStride) + PathologicalDslConstants.StaticOffset) % 53;
        var output = input[baseIndex] + uniformBias;
        var texture = MathF.Sin(output) + MathF.Cos(output * 0.5f);
        var sampler = 0.0f;

        for (var v = 0; v < PathologicalDslConstants.Tile; v++)
        {
            var idx = (baseIndex + (v * dynamicStride) + ((i + v) % localScale)) % input.Length;
            sampler += input[idx] * (v + 1);
        }

        for (var v = 0; v < 3; v++)
        {
            if ((v & 1) == 0)
            {
                continue;
            }

            sampler += v * 0.125f;
            if (sampler > 12.0f)
            {
                break;
            }
        }

        var shaped = ShapeReference(output, texture, sampler, i);
        var whileIndex = 0;
        while (whileIndex < layout + 1)
        {
            shaped += 0.1f * (whileIndex + 1);
            whileIndex++;
        }

        var doIndex = 0;
        do
        {
            shaped -= 0.05f;
            doIndex++;
        }
        while (doIndex < 2);

        if (shaped > 0.0f)
        {
            var buffer = shaped * 0.5f;
            shaped = buffer;
        }
        else
        {
            var buffer = -shaped + 0.25f;
            shaped = buffer;
        }

        var outputIndex = (i * PathologicalDslConstants.OutputStride) + (i % 2);
        return new PathologicalPreShared(shaped, texture, sampler, outputIndex);
    }

    private static float ShapeReference(float output, float texture, float sampler, int input)
    {
        var uniform = InnerCurveReference(output + texture, sampler);
        return (input & 1) == 0
            ? MathF.Max(uniform, output - sampler)
            : MathF.Min(uniform, texture + sampler);
    }

    private static float InnerCurveReference(float value, float bias)
    {
        var result = (value * 0.5f) + (bias * PathologicalDslConstants.StaticScale);
        return result > 2.0f ? result - 1.0f : result + 0.25f;
    }

    private static void AssertNear(float4 expected, float4 actual, float tolerance)
    {
        AssertNear(expected.X, actual.X, tolerance);
        AssertNear(expected.Y, actual.Y, tolerance);
        AssertNear(expected.Z, actual.Z, tolerance);
        AssertNear(expected.W, actual.W, tolerance);
    }

    private static void AssertNear(float expected, float actual, float tolerance)
        => Assert.InRange(MathF.Abs(actual - expected), 0, tolerance);

    private readonly record struct PathologicalExpected(float4[] Vectors, float[] Scalars);

    private readonly record struct PathologicalPreShared(float Shaped, float Texture, float Sampler, int OutputIndex);
}

/// <summary>
/// Nested std430 input record used by the basic DSL stress dispatch.
/// </summary>
[GpuStruct]
public partial struct DslStressInner
{
    /// <summary>
    /// Direction vector packed as a std430 vec3 field.
    /// </summary>
    public float3 Direction;

    /// <summary>
    /// Scalar that must occupy the vec3 padding slot at offset 12.
    /// </summary>
    public float Weight;
}

/// <summary>
/// Input struct that combines nested structs, vectors, scalars, and integer fields.
/// </summary>
[GpuStruct]
public partial struct DslStressRecord
{
    /// <summary>
    /// Nested std430 struct value.
    /// </summary>
    public DslStressInner Inner;

    /// <summary>
    /// Two-component coordinate used by matrix/vector expressions.
    /// </summary>
    public float2 Uv;

    /// <summary>
    /// Dynamic loop bound read from GPU memory.
    /// </summary>
    public int IterationLimit;

    /// <summary>
    /// Scalar bias consumed by nested callables.
    /// </summary>
    public float Bias;

    /// <summary>
    /// Integer flags used by logical and bitwise expressions.
    /// </summary>
    public int Flags;
}

/// <summary>
/// Output struct that proves nested field writes and std430 readback packing.
/// </summary>
[GpuStruct]
public partial struct DslStressResult
{
    /// <summary>
    /// Vector result written through a struct field lvalue.
    /// </summary>
    public float4 Color;

    /// <summary>
    /// Normalized vector result packed as a std430 vec3 field.
    /// </summary>
    public float3 Normal;

    /// <summary>
    /// Scalar written into the vec3 padding slot.
    /// </summary>
    public float Score;

    /// <summary>
    /// Integer loop-bound result written after the packed vector fields.
    /// </summary>
    public int Steps;
}

/// <summary>
/// Exercises the basic DSL surface through the typed EasyGPU lowering path.
/// </summary>
[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BasicDslStressKernel(
    ReadOnlyBuffer<DslStressRecord> input,
    ReadWriteBuffer<DslStressResult> output,
    ReadWriteBuffer<float> scalarOutput,
    ReadWriteBuffer<int> intOutput,
    Uniform<int> globalIterations,
    Uniform<float> globalBias) : IKernel1D
{
    /// <summary>
    /// Runs the stress program for one logical element.
    /// </summary>
    public void Execute()
    {
        int i = ThreadIds.X;
        DslStressRecord record = input[i];
        float2x2 transform = new float2x2(new float2(1.0f, 2.0f), new float2(3.0f, 4.0f));
        float2 transformed = transform * new float2(record.Uv.Y, record.Uv.X);
        float3 mixed = new float3(transformed.X, transformed.Y, record.Inner.Direction.Z) + record.Inner.Direction;
        float dot = ShaderMath.Dot(mixed, new float3(1.0f, 0.5f, -1.0f));
        float3 cross = ShaderMath.Cross(record.Inner.Direction, new float3(0.0f, 1.0f, 0.0f));
        float shaped = Shape(dot + record.Inner.Weight, record.Bias + globalBias.Value);
        int limit = record.IterationLimit + globalIterations.Value;
        float accumulator = 0.0f;

        for (int k = 0; k < limit; k++)
        {
            if ((k % 2) == 1)
            {
                continue;
            }

            accumulator += k * 0.5f;
            if (accumulator > 3.0f)
            {
                break;
            }
        }

        int whileIndex = 0;
        while (whileIndex < record.IterationLimit)
        {
            accumulator += 0.25f;
            whileIndex++;
        }

        int doIndex = 0;
        do
        {
            accumulator += 0.125f;
            doIndex++;
        }
        while (doIndex < 2);

        bool keep = (((record.Flags & 1) == 1) || shaped > 0.0f) && !(record.Flags == 0);
        float score = keep ? shaped + accumulator : -shaped;
        float clamped = ShaderMath.Clamp(score, -32.0f, 32.0f);
        float3 normal = ShaderMath.Normalize(cross + new float3(0.01f, 0.0f, 0.0f));

        output[i].Color = new float4(normal.XY, clamped, 1.0f);
        output[i].Normal = normal;
        output[i].Score = clamped;
        output[i].Steps = limit;
        scalarOutput[i] = clamped;
        intOutput[i] = ((int)clamped) + ((record.Flags << 1) ^ (i & 1));
    }

    /// <summary>
    /// Outer callable that invokes another callable and includes callable-local control flow.
    /// </summary>
    [Callable]
    private static float Shape(float value, float bias)
    {
        float curved = InnerCurve(value, bias);
        if (curved > 4.0f)
        {
            return curved - 1.0f;
        }

        return curved + 2.0f;
    }

    /// <summary>
    /// Inner callable used to verify nested callable lowering.
    /// </summary>
    [Callable]
    private static float InnerCurve(float value, float bias)
    {
        return (value * 0.75f) + bias;
    }
}

internal static class PathologicalDslConstants
{
    public const int Tile = 4;
    public const int InputStride = 7;
    public const int OutputStride = 2;
    public static readonly int StaticOffset = 3;
    public static readonly float StaticScale = 0.25f;
}

/// <summary>
/// Intentionally combines pathological but supported C# shader syntax in one kernel.
/// </summary>
[Kernel]
[ThreadGroupSize(4, 1, 1)]
public readonly partial struct PathologicalDslTortureKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float4> vectorOutput,
    ReadWriteBuffer<float> scalarOutput,
    Uniform<int> dynamicStride,
    Uniform<float> uniformBias) : IKernel1D
{
    public void Execute()
    {
        var @in = ThreadIds.X;
        var layout = LocalIds.X;
        var shared = new SharedMemory<float>(PathologicalDslConstants.Tile);
        const int localScale = 2;
        var baseIndex = ((@in * PathologicalDslConstants.InputStride) + PathologicalDslConstants.StaticOffset) % 53;
        float output = input[baseIndex] + uniformBias.Value;
        float texture = ShaderMath.Sin(output) + ShaderMath.Cos(output * 0.5f);
        float sampler = 0.0f;

        for (var v = 0; v < PathologicalDslConstants.Tile; v = v + 1)
        {
            var idx = (baseIndex + (v * dynamicStride.Value) + ((@in + v) % localScale)) % 64;
            sampler = sampler + (input[idx] * (v + 1));
        }

        for (var v = 0; v < 3; v = v + 1)
        {
            if ((v & 1) == 0)
            {
                continue;
            }

            sampler = sampler + (v * 0.125f);
            if (sampler > 12.0f)
            {
                break;
            }
        }

        float shaped = Shape(output, texture, sampler, @in);
        int whileIndex = 0;
        while (whileIndex < layout + 1)
        {
            shaped = shaped + (0.1f * (whileIndex + 1));
            whileIndex++;
        }

        int doIndex = 0;
        do
        {
            shaped = shaped - 0.05f;
            doIndex++;
        }
        while (doIndex < 2);

        if (shaped > 0.0f)
        {
            float buffer = shaped * 0.5f;
            shaped = buffer;
        }
        else
        {
            float buffer = -shaped + 0.25f;
            shaped = buffer;
        }

        shared[layout] = shaped;
        GpuBarrier.Workgroup();
        float neighbor = shared[(PathologicalDslConstants.Tile - 1) - layout];
        GpuBarrier.Workgroup();

        float uniform = shaped + (neighbor * 0.125f);
        float2 xy = new float2(uniform, texture);
        float2x2 transform = new float2x2(new float2(1.0f, 0.25f), new float2(-0.5f, 0.75f));
        float2 transformed = transform * xy;
        float3 normal = ShaderMath.Normalize(new float3(transformed.X, transformed.Y, sampler) + new float3(0.01f, 0.02f, 0.03f));
        float clamped = ShaderMath.Clamp(uniform + ShaderMath.Dot(normal, new float3(0.25f, -0.5f, 0.75f)), -32.0f, 32.0f);
        float4 packed = new float4(normal.XY, clamped, 1.0f);
        var @out = (@in * PathologicalDslConstants.OutputStride) + (@in % 2);

        vectorOutput[@in] = packed;
        scalarOutput[@out] = packed.Z + (@out % 3);
    }

    [Callable]
    private static float Shape(float output, float texture, float sampler, int input)
    {
        float uniform = InnerCurve(output + texture, sampler);
        if ((input & 1) == 0)
        {
            return ShaderMath.Max(uniform, output - sampler);
        }

        return ShaderMath.Min(uniform, texture + sampler);
    }

    [Callable]
    private static float InnerCurve(float value, float bias)
    {
        float result = (value * 0.5f) + (bias * PathologicalDslConstants.StaticScale);
        if (result > 2.0f)
        {
            return result - 1.0f;
        }

        return result + 0.25f;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test kernel definitions
// ═══════════════════════════════════════════════════════════════════════

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DslCopyKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DslLocalVarKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        // Uses local variable 'i' as buffer index
        output[i] = input[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DslIfElseKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        // Real if/else with comparison, NOT raw string injection
        if (input[i] > 0)
            output[i] = input[i];
        else
            output[i] = 0;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DslIntrinsicKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ShaderMath.Sin(input[i]) + ShaderMath.Sqrt(input[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DslForLoopKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        // For-loop with break — tests for-loop + if + break lowering
        for (;;)
        {
            output[i] = input[i] * 2;
            break;
        }
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DslTernaryKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DslCompoundAddKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] += input[i];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct DslFloat2Kernel(
    ReadOnlyBuffer<float2> a,
    ReadOnlyBuffer<float2> b,
    ReadWriteBuffer<float2> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = a[i] + b[i];
    }
}


public class JuliaSetBasicTests
{
    [Fact]
    public void Float3_Construction_CompilesAndDispatches()
    {
        using var output = GPU.CreateBuffer<float3>(4);
        GPU.Dispatch(new Float3ConstructKernel(output.AsReadWrite()), 4);
        Assert.Equal([
            new float3(1, 2, 3),
            new float3(1, 2, 3),
            new float3(1, 2, 3),
            new float3(1, 2, 3)
        ], output.ToArray());
    }

    [Fact]
    public void JuliaSetKernel_WritesMeaningfulImageThroughTypedEasyGpu()
    {
        const int width = 16;
        const int height = 16;
        var glsl = ShaderInspection.GetGLSL<JuliaSetKernel>();
        Assert.Contains("gl_GlobalInvocationID", glsl, StringComparison.Ordinal);
        Assert.Contains("while", glsl, StringComparison.Ordinal);
        Assert.Contains("layout(push_constant) uniform EasyGPUUniformBlock", glsl, StringComparison.Ordinal);
        Assert.Contains("int u0;", glsl, StringComparison.Ordinal);
        Assert.Contains("int u1;", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var output = GPU.CreateBuffer<int>(width * height, BufferAccess.ReadWrite);
        var path = GPU.DispatchAndGetPath(
            new JuliaSetKernel(output.AsReadWrite(), new Uniform<int>(width), new Uniform<int>(height)),
            new int2(width, height));
        var pixels = output.ToArray();
        var imagePath = Path.Combine(Path.GetTempPath(), $"feather-julia-{Guid.NewGuid():N}.tga");

        try
        {
            WriteArgbTga(imagePath, pixels, width, height);

            Assert.Equal(DispatchPath.TypedEasyGpu, path);
            Assert.True(new FileInfo(imagePath).Length > 18);
            Assert.True(pixels.Count(pixel => (pixel & unchecked((int)0x00FFFFFF)) != 0) > pixels.Length / 8);
            Assert.True(pixels.Select(pixel => pixel & unchecked((int)0x00FFFFFF)).Distinct().Count() > 8);
        }
        finally
        {
            File.Delete(imagePath);
        }
    }

    private static void WriteArgbTga(string path, ReadOnlySpan<int> pixels, int width, int height)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var stream = File.Create(path);
        Span<byte> header = stackalloc byte[18];
        header[2] = 2;
        header[12] = (byte)width;
        header[13] = (byte)(width >> 8);
        header[14] = (byte)height;
        header[15] = (byte)(height >> 8);
        header[16] = 32;
        header[17] = 0x28;
        stream.Write(header);

        var bytes = MemoryMarshal.AsBytes(pixels);
        var encoded = new byte[checked(width * height * 4)];
        for (var i = 0; i < width * height; i++)
        {
            var source = i * 4;
            encoded[source] = bytes[source];
            encoded[source + 1] = bytes[source + 1];
            encoded[source + 2] = bytes[source + 2];
            encoded[source + 3] = bytes[source + 3];
        }

        stream.Write(encoded);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct Float3ConstructKernel(
    ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = new float3(1.0f, 2.0f, 3.0f);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct JuliaSetKernel(
    ReadWriteBuffer<int> output,
    Uniform<int> width,
    Uniform<int> height) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        int imageWidth = width.Value;
        int imageHeight = height.Value;
        float x = ((p.X * 3.0f) / (imageWidth - 1)) - 1.5f;
        float y = ((p.Y * 3.0f) / (imageHeight - 1)) - 1.5f;
        float zx = x;
        float zy = y;
        float cx = -0.7f;
        float cy = 0.27f;
        int iter = 0;
        while (iter < 256)
        {
            float nx = zx * zx - zy * zy + cx;
            float ny = 2.0f * zx * zy + cy;
            zx = nx;
            zy = ny;
            if (zx * zx + zy * zy > 4.0f)
            {
                break;
            }

            iter++;
        }

        int color = iter < 256 ? (iter * 8) % 256 : 0;
        output[p.Y * imageWidth + p.X] = color | (color << 8) | (color << 16) | (255 << 24);
    }
}

public class CallableCoverageTests
{
    [Fact]
    public void ScalarCallable_CompilesAndDispatches()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new CallableScalarKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        // callable returns input * 2 + 1
        Assert.Equal([3, 5, 7, 9], output.ToArray());
    }

    [Fact]
    public void Callable_GeneratesValidGlsl()
    {
        var glsl = ShaderInspection.GetGLSL<NestedCallableKernel>();
        const string innerName = "global__Feather_Integration_Tests_NestedCallableKernel_Inner_";
        const string outerName = "global__Feather_Integration_Tests_NestedCallableKernel_Outer_";
        const string innerDefinitionPrefix = "float global__Feather_Integration_Tests_NestedCallableKernel_Inner_";
        const string outerDefinitionPrefix = "float global__Feather_Integration_Tests_NestedCallableKernel_Outer_";
        var mainStart = glsl.IndexOf("void main()", StringComparison.Ordinal);
        var innerBodyStart = mainStart < 0 ? -1 : glsl.IndexOf(innerDefinitionPrefix, mainStart, StringComparison.Ordinal);
        var outerBodyStart = mainStart < 0 ? -1 : glsl.IndexOf(outerDefinitionPrefix, mainStart, StringComparison.Ordinal);
        var outerCallStart = mainStart < 0 ? -1 : glsl.IndexOf(outerName, mainStart, StringComparison.Ordinal);
        var innerCallStart = outerCallStart < 0 ? -1 : glsl.IndexOf(innerName, outerCallStart, StringComparison.Ordinal);

        Assert.True(mainStart >= 0, "GLSL must contain an entry-point main function.");
        Assert.True(innerBodyStart > mainStart, "Callable definitions should be emitted after main with forward declarations.");
        Assert.True(outerBodyStart > mainStart, "All callable definitions should be emitted after main with forward declarations.");
        Assert.True(outerCallStart > mainStart && outerCallStart < innerBodyStart, "Main should call the forwarded outer callable before callable definitions.");
        Assert.True(innerCallStart > outerCallStart && innerCallStart < innerBodyStart, "Main should pass the forwarded inner callable result into the outer call.");
        Assert.Contains(innerDefinitionPrefix, glsl, StringComparison.Ordinal);
        Assert.Contains(outerDefinitionPrefix, glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoArgCallable_CompilesAndDispatches()
    {
        using var input1 = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var input2 = GPU.CreateBuffer<float>([10, 20, 30, 40]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new TwoArgCallableKernel(input1.AsReadOnly(), input2.AsReadOnly(), output.AsReadWrite()), 4);
        // AddMul(x, y) = x * y + x → [1*10+1=11, 2*20+2=42, 3*30+3=93, 4*40+4=164]
        Assert.Equal([11f, 42f, 93f, 164f], output.ToArray());
    }

    [Fact]
    public void NestedCallable_CompilesAndDispatches()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new NestedCallableKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        // Outer(Inner(x)) where Inner(x)=x*2, Outer(x)=x+1 → (x*2)+1 → [3,5,7,9]
        Assert.Equal([3f, 5f, 7f, 9f], output.ToArray());
    }

    [Fact]
    public void ExpressionBodyCallable_Dispatches()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new ExpressionCallableKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        // Scale(x) = x * 3.0f → [3, 6, 9, 12]
        Assert.Equal([3f, 6f, 9f, 12f], output.ToArray());
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct CallableScalarKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Transform(input[i]);
    }

    [Callable]
    private static float Transform(float x)
    {
        return x * 2.0f + 1.0f;
    }
}

// ── Two-arg callable ─────────────────────────────────────────────────────

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TwoArgCallableKernel(
    ReadOnlyBuffer<float> a,
    ReadOnlyBuffer<float> b,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = AddMul(a[i], b[i]);
    }

    [Callable]
    private static float AddMul(float x, float y)
    {
        return x * y + x;
    }
}

// ── Nested callable ──────────────────────────────────────────────────────

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct NestedCallableKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Outer(Inner(input[i]));
    }

    [Callable]
    private static float Inner(float x)
    {
        return x * 2.0f;
    }

    [Callable]
    private static float Outer(float x)
    {
        return x + 1.0f;
    }
}

// ── Expression-body callable (=> syntax) ────────────────────────────────

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct ExpressionCallableKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Scale(input[i]);
    }

    [Callable]
    private static float Scale(float x) => x * 3.0f;
}

// ── Sampled texture + sampler ────────────────────────────────────────────

public class TextureSampleCoverageTests
{
    [Fact]
    public void TextureSample_GeneratesTypedEasyGpuGlsl()
    {
        var glsl = ShaderInspection.GetGLSL<TextureSampleKernel>();

        Assert.Contains("sampler2D", glsl, StringComparison.Ordinal);
        Assert.Contains("texture(", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void TextureSampleLevel_GeneratesTypedEasyGpuGlsl()
    {
        var glsl = ShaderInspection.GetGLSL<TextureSampleLevelKernel>();

        Assert.Contains("sampler2D", glsl, StringComparison.Ordinal);
        Assert.Contains("textureLod(", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void TextureSample_IrContainsNodeKind()
    {
        var ir = ShaderInspection.GetIR<TextureSampleKernel>();
        Assert.Contains("0C", ir, StringComparison.Ordinal); // 12 = TextureSample
    }

    [Fact]
    public void TextureSample_DispatchesThroughTypedEasyGpuAndReadsBack()
    {
        using var texture = GPU.CreateTexture2D<Rgba32, Rgba32>(1, 1, PixelFormat.Rgba8, TextureAccess.Sampled);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var output = GPU.CreateBuffer<float>(1);
        texture.Upload([new Rgba32(64, 0, 0, 255)]);

        var path = GPU.DispatchAndGetPath(
            new TextureSampleKernel(texture.AsSampled(), sampler, output.AsReadWrite()),
            1);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        AssertNear(64.0f / 255.0f, output.ToArray()[0]);
    }

    [Fact]
    public void TextureSampleLevel_IrContainsNodeKind()
    {
        var ir = ShaderInspection.GetIR<TextureSampleLevelKernel>();
        Assert.Contains("0D", ir, StringComparison.Ordinal); // 13 = TextureSampleLevel
    }

    [Fact]
    public void TextureSampleLevel_DispatchesThroughTypedEasyGpuAndReadsBack()
    {
        using var texture = GPU.CreateTexture2D<Rgba32, Rgba32>(1, 1, PixelFormat.Rgba8, TextureAccess.Sampled);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var output = GPU.CreateBuffer<float>(1);
        texture.Upload([new Rgba32(128, 0, 0, 255)]);

        var path = GPU.DispatchAndGetPath(
            new TextureSampleLevelKernel(texture.AsSampled(), sampler, output.AsReadWrite()),
            1);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        AssertNear(128.0f / 255.0f, output.ToArray()[0]);
    }

    [Fact]
    public void TextureSample_DispatchesSupportedFormatMatrixThroughTypedEasyGpu()
    {
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var r8 = GPU.CreateTexture2D<byte, float4>(1, 1, PixelFormat.R8, TextureAccess.Sampled);
        using var rg8 = GPU.CreateTexture2D<SampleRg8, float4>(1, 1, PixelFormat.Rg8, TextureAccess.Sampled);
        using var r32 = GPU.CreateTexture2D<float, float4>(1, 1, PixelFormat.R32Float, TextureAccess.Sampled);
        using var rgba32 = GPU.CreateTexture2D<float4, float4>(1, 1, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var output = GPU.CreateBuffer<float>(4);

        r8.Upload([byte.CreateChecked(64)]);
        rg8.Upload([new SampleRg8(96, 128)]);
        r32.Upload([0.75f]);
        rgba32.Upload([new float4(1.25f, 2.0f, 3.0f, 4.0f)]);

        var path = GPU.DispatchAndGetPath(
            new TextureSampleFormatMatrixKernel(
                r8.AsSampled(),
                rg8.AsSampled(),
                r32.AsSampled(),
                rgba32.AsSampled(),
                sampler,
                output.AsReadWrite()),
            1);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        var values = output.ToArray();
        AssertNear(64.0f / 255.0f, values[0]);
        AssertNear((96.0f + 128.0f) / 255.0f, values[1]);
        AssertNear(0.75f, values[2]);
        AssertNear(1.25f + 4.0f, values[3]);
    }

    [Fact]
    public void TextureSampleLevel_DispatchesRgba32FloatThroughTypedEasyGpu()
    {
        using var texture = GPU.CreateTexture2D<float4, float4>(1, 1, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var output = GPU.CreateBuffer<float>(1);
        texture.Upload([new float4(2.5f, 0.0f, 0.0f, 1.0f)]);

        var path = GPU.DispatchAndGetPath(
            new TextureSampleLevelFloat4Kernel(texture.AsSampled(), sampler, output.AsReadWrite()),
            1);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        AssertNear(2.5f, output.ToArray()[0]);
    }

    private static void AssertNear(float expected, float actual, float tolerance = 1e-4f)
        => Assert.InRange(MathF.Abs(actual - expected), 0, tolerance);
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureSampleKernel(
    SampledTexture2D<Rgba32> input,
    SamplerState sampler,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 uv = new float2(0.5f, 0.5f);
        output[i] = input.Sample(sampler, uv).R;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureSampleLevelKernel(
    SampledTexture2D<Rgba32> input,
    SamplerState sampler,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 uv = new float2(0.5f, 0.5f);
        output[i] = input.SampleLevel(sampler, uv, 0.0f).R;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureSampleFormatMatrixKernel(
    SampledTexture2D<float4> r8,
    SampledTexture2D<float4> rg8,
    SampledTexture2D<float4> r32,
    SampledTexture2D<float4> rgba32,
    SamplerState sampler,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        float2 uv = new float2(0.5f, 0.5f);
        float4 r8Pixel = r8.Sample(sampler, uv);
        float4 rg8Pixel = rg8.Sample(sampler, uv);
        float4 r32Pixel = r32.Sample(sampler, uv);
        float4 rgba32Pixel = rgba32.Sample(sampler, uv);
        output[0] = r8Pixel.R;
        output[1] = rg8Pixel.R + rg8Pixel.G;
        output[2] = r32Pixel.R;
        output[3] = rgba32Pixel.R + rgba32Pixel.A;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureSampleLevelFloat4Kernel(
    SampledTexture2D<float4> input,
    SamplerState sampler,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 uv = new float2(0.5f, 0.5f);
        output[i] = input.SampleLevel(sampler, uv, 0.0f).R;
    }
}

[GpuStruct]
public readonly partial record struct SampleRg8(byte R, byte G);
