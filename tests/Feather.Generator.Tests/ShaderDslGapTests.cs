using System.Collections.Immutable;
using Feather.Generators;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace Feather.Generator.Tests;

/// <summary>
/// Minimal reproductions for shader-DSL gaps reported from the PAR project:
/// local GpuStruct field mutation, [InlineArray] fields in shader-local structs,
/// [Callable] ref/out parameters, ReadOnlyBuffer&lt;T&gt; with generic struct element type,
/// generic type-parameter local variables, and a generic [Callable] calling another generic
/// [Callable] (the outer one discovered through a concrete construction at the kernel call
/// site, which used to crash type-argument substitution with an InvalidOperationException).
/// </summary>
public class ShaderDslGapTests
{
    [Fact]
    public void LocalGpuStructFieldMutationLowers()
    {
        var (diagnostics, generated) = RunGenerator("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct Counter
            {
                public int Count;
                public float Weight;

                public Counter(int count, float weight)
                {
                    Count = count;
                    Weight = weight;
                }
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct MutateKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    var s = new Counter(3, 0.5f);
                    s.Count = 0;
                    s.Weight = s.Weight + 1;
                    output[ThreadIds.X] = s.Count + s.Weight;
                }
            }
            """);

        var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(errors.Length == 0, string.Join("\n", errors.Select(e => e.ToString())) + "\n--- generated ---\n" + generated);
    }

    [Fact]
    public void LocalGpuStructInstanceMethodThisMutationLowers()
    {
        var (diagnostics, generated) = RunGenerator("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct Accumulator
            {
                public float Total;

                [Callable]
                public void Reset()
                {
                    Total = 0;
                }

                [Callable]
                public void Add(float value)
                {
                    Total = Total + value;
                }
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct MethodKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    var acc = new Accumulator();
                    acc.Reset();
                    acc.Add(2);
                    output[ThreadIds.X] = acc.Total;
                }
            }
            """);

        var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(errors.Length == 0, string.Join("\n", errors.Select(e => e.ToString())) + "\n--- generated ---\n" + generated);
    }

    [Fact]
    public void InlineArrayFieldInShaderLocalStructLowers()
    {
        var (diagnostics, generated) = RunGenerator("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;
            using System.Runtime.CompilerServices;

            namespace Scratch;

            [InlineArray(4)]
            public struct Float4Lanes
            {
                private float4 _element0;
            }

            [GpuStruct]
            public partial struct LanePacket
            {
                public Float4Lanes Lanes;
                public int Count;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct LaneKernel(ReadWriteBuffer<float4> output) : IKernel1D
            {
                public void Execute()
                {
                    var packet = new LanePacket();
                    packet.Lanes[0] = new float4(1, 2, 3, 4);
                    packet.Lanes[1] = new float4(5, 6, 7, 8);
                    packet.Count = 2;
                    output[ThreadIds.X] = packet.Lanes[0] + packet.Lanes[packet.Count - 1];
                }
            }
            """);

        var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(errors.Length == 0, string.Join("\n", errors.Select(e => e.ToString())) + "\n--- generated ---\n" + generated);
    }

    [Fact]
    public void CallableRefOutParametersLower()
    {
        var (diagnostics, generated) = RunGenerator("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct SwapKernel(ReadWriteBuffer<float2> output) : IKernel1D
            {
                public void Execute()
                {
                    float2 a = new float2(1, 2);
                    float2 b = new float2(3, 4);
                    Swap(ref a, ref b);
                    float hi;
                    TakeMax(a, b, out hi);
                    output[ThreadIds.X] = a + b + new float2(hi, hi);
                }

                [Callable]
                private static void Swap(ref float2 x, ref float2 y)
                {
                    float2 t = x;
                    x = y;
                    y = t;
                }

                [Callable]
                private static void TakeMax(float2 x, float2 y, out float result)
                {
                    result = x.X > y.X ? x.X : y.X;
                }
            }
            """);

        var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(errors.Length == 0, string.Join("\n", errors.Select(e => e.ToString())) + "\n--- generated ---\n" + generated);
    }

    [Fact]
    public void GenericCallableTakingReadOnlyBufferOfTypeParameterDoesNotCrash()
    {
        var (diagnostics, generated) = RunGenerator("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            public interface IWeighted
            {
                float4 Value { get; }
            }

            [GpuStruct]
            public readonly partial record struct Weighted(float4 Value) : IWeighted;

            [ShaderLibrary]
            public static class Queries
            {
                [Callable]
                public static float4 FirstValue<TShape>(ReadOnlyBuffer<TShape> shapes, float fallback)
                    where TShape : unmanaged, IWeighted
                {
                    TShape first = shapes[0];
                    return first.Value + new float4(fallback, fallback, fallback, fallback);
                }
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct WeightedKernel(ReadOnlyBuffer<Weighted> input, ReadWriteBuffer<float4> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = Queries.FirstValue(input, 0.5f);
                }
            }
            """);

        var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(errors.Length == 0, string.Join("\n", errors.Select(e => e.ToString())) + "\n--- generated ---\n" + generated);
    }

    [Fact]
    public void GenericTypeParameterLocalVariableLowers()
    {
        var (diagnostics, generated) = RunGenerator("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct PassthroughKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = Echo(input[ThreadIds.X]);
                }

                [Callable]
                private static T Echo<T>(T value) where T : unmanaged
                {
                    T local = value;
                    return local;
                }
            }
            """);

        var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(errors.Length == 0, string.Join("\n", errors.Select(e => e.ToString())) + "\n--- generated ---\n" + generated);
    }

    [Fact]
    public void GenericCallableCallingGenericCallableMonomorphizes()
    {
        var (diagnostics, generated) = RunGenerator("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            public interface IConvex2D
            {
                float Sdf(float2 x);
                bool IsEmitter();
                float3 Emission();
                float AlbedoFactor { get; }
            }

            [GpuStruct]
            public readonly partial record struct Box(float2 Center, float2 Half, float3 Emit, float AlbedoFactor) : IConvex2D
            {
                [Callable]
                public float Sdf(float2 x) => 0.0f;
                [Callable]
                public bool IsEmitter() => true;
                [Callable]
                public float3 Emission() => Emit;
            }

            [GpuStruct]
            public readonly partial record struct SceneHit(float3 Emission, float AlbedoFactor, float Distance)
            {
                [Callable]
                public SceneHit Min(SceneHit other) => other.Distance < Distance ? other : this;
            }

            [ShaderLibrary]
            public static class SceneQueries
            {
                [Callable]
                public static SceneHit Nearest<TShape>(ReadOnlyBuffer<TShape> shapes, int count, float2 x)
                    where TShape : unmanaged, IConvex2D
                {
                    SceneHit best = new SceneHit(new float3(0.0f), 1.0f, 1e9f);
                    for (int i = 0; i < count; i++)
                    {
                        best = NearestOne(best, shapes[i], x);
                    }
                    return best;
                }

                [Callable]
                public static SceneHit NearestOne<TShape>(SceneHit best, TShape shape, float2 x)
                    where TShape : unmanaged, IConvex2D
                {
                    float sdf = shape.Sdf(x);
                    if (sdf < best.Distance)
                    {
                        return new SceneHit(shape.IsEmitter() ? shape.Emission() : new float3(0.0f), shape.AlbedoFactor, sdf);
                    }
                    return best;
                }
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct QueryKernel(ReadOnlyBuffer<Box> boxes, ReadWriteBuffer<float4> output) : IKernel1D
            {
                public void Execute()
                {
                    SceneHit hit = SceneQueries.Nearest(boxes, 1, new float2(0.0f));
                    output[ThreadIds.X] = new float4(hit.Emission, hit.Distance);
                }
            }
            """);

        var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(errors.Length == 0, string.Join("\n", errors.Select(e => e.ToString())) + "\n--- generated ---\n" + generated);
        Assert.Contains("QueryKernel", generated);
    }

    private static (ImmutableArray<Diagnostic> Diagnostics, string GeneratedSources) RunGenerator(string source)
    {
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(assembly => !assembly.IsDynamic && !string.IsNullOrWhiteSpace(assembly.Location))
            .Select(assembly => MetadataReference.CreateFromFile(assembly.Location))
            .Concat([MetadataReference.CreateFromFile(typeof(KernelAttribute).Assembly.Location)])
            .Distinct()
            .ToArray();

        var compilation = CSharpCompilation.Create(
            "Scratch",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true));

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var compileErrors = compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        if (compileErrors.Length > 0)
        {
            throw new InvalidOperationException("test source does not compile:\n" + string.Join("\n", compileErrors.Select(e => e.ToString())));
        }

        var generated = string.Join("\n// ----\n",
            outputCompilation.SyntaxTrees
                .Where(tree => tree.FilePath.Contains("Feather", StringComparison.Ordinal))
                .Select(tree => tree.ToString()));
        return (diagnostics, generated);
    }
}
