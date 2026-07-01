using System.Collections.Immutable;
using Feather.Generators;
using Feather.Generators.Model;
using Feather.Interop;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Feather.Generator.Tests;

public class GeneratorAndAnalyzerTests
{
    [Fact]
    public void GeneratorEmitsKernelDescriptorForAnnotatedKernel()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(DefaultThreadGroupSizes.X)]
            public readonly partial struct AddKernel(ReadOnlyBuffer<float> a, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = a[i];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("AddKernel.Feather.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();
        Assert.Contains("IGeneratedKernel<AddKernel>", source);
        Assert.Contains("KernelDimension.One", source);
        Assert.Contains("ResourceKind.Buffer", source);
        Assert.Contains("ResourceAccess.ReadWrite", source);

        var module = ReadGeneratedIr(source);

        Assert.Equal("AddKernel", module.EntryPoint);
        Assert.Equal(FeatherIrShaderKind.Compute1D, module.ShaderKind);
        Assert.Equal(2, module.Resources.Count);
        Assert.Contains(module.Resources, resource => resource.Name == "a" && resource.Kind == FeatherIrResourceKind.Buffer && resource.Access == FeatherIrResourceAccess.Read);
        Assert.Contains(module.Resources, resource => resource.Name == "output" && resource.Kind == FeatherIrResourceKind.Buffer && resource.Access == FeatherIrResourceAccess.ReadWrite);
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.LocalDeclaration && instruction.Operand.Contains("ThreadIds.X", StringComparison.Ordinal));
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.Assignment
            && instruction.OperandKind == FeatherIrOperandKind.ElementwiseAssignment
            && instruction.Operand == "ASSIGN1|output|i|copy|a|");
        var assignment = Assert.Single(module.ElementwiseAssignments);
        Assert.Equal(1u, assignment.DestinationBinding);
        Assert.Equal(0u, assignment.LeftBinding);
        Assert.Equal(FeatherIrAssignmentOperation.Copy, assignment.Operation);
        Assert.Equal(FeatherIrAssignmentOperandKind.None, assignment.RightOperandKind);
        Assert.Equal("i", assignment.Index);
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.ResourceAccess
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "RESOURCE1|output|i");
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.ResourceAccess
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "RESOURCE1|a|i");
    }

    [Fact]
    public void GeneratorUsesEntryAttributeWhenPresent()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct AddKernel(ReadOnlyBuffer<float> a, ReadWriteBuffer<float> output) : IKernel1D
            {
                [Entry]
                public void Run()
                {
                    int i = ThreadIds.X;
                    output[i] = a[i] + 1.0f;
                }

                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = -1000.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("AddKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Single(module.ElementwiseExpressionAssignments);
        Assert.DoesNotContain(module.Instructions, instruction => instruction.Operand.Contains("-1000", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorDoesNotEmitComputeKernelWhenTypedIrLoweringFails()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = global::System.MathF.Sqrt(1.0f);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0027");
        Assert.Contains("System.MathF.Sqrt", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree =>
            tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("length")]
    public void GeneratorRejectsInvalidSharedMemoryLengths(string lengthExpression)
    {
        var source = $$"""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadSharedKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int length = 4;
                    var shared = new SharedMemory<float>({{lengthExpression}});
                    output[ThreadIds.X] = 0;
                }
            }
            """;
        var compilation = CreateCompilation(source);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0027");
        Assert.Contains("shared memory length", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree =>
            tree.FilePath.EndsWith("BadSharedKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsInvalidAtomicTarget()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadAtomicKernel : IKernel1D
            {
                public void Execute()
                {
                    int value = ThreadIds.X;
                    _ = GpuAtomic.Add(ref value, 1);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0027");
        Assert.Contains("atomic target for 'Add'", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.Contains("local 'value'", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree =>
            tree.FilePath.EndsWith("BadAtomicKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("ReadWriteBuffer<byte>", "byte")]
    [InlineData("ReadWriteBuffer<sbyte>", "sbyte")]
    [InlineData("ReadWriteBuffer<short>", "short")]
    [InlineData("ReadWriteBuffer<ushort>", "ushort")]
    public void GeneratorRejectsNarrowScalarComputeBuffers(string bufferType, string expectedType)
    {
        var compilation = CreateCompilation($$"""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel({{bufferType}} output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = default;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0004");
        Assert.Contains(expectedType, diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree =>
            tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorEmitsGraphicsPipelineContractForAnnotatedShaders()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [VertexShader]
            public readonly partial struct TestVS(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
            {
                public float4 Execute()
                {
                    return vertices[0];
                }
            }

            [FragmentShader]
            public readonly partial struct TestFS(SamplerState sampler) : IFragmentShader<float4>
            {
                public float4 Execute(float4 input)
                {
                    return input;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("TestVS_TestFS.Feather.Graphics.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();

        Assert.Contains("IGeneratedGraphicsPipeline<global::Scratch.TestVS, global::Scratch.TestFS, global::Feather.Math.float4>", source);
        Assert.Contains("GraphicsPipelineDescriptor", source);
        Assert.Contains(".VertexIR => new byte[]", source);
        Assert.Contains(".FragmentIR => new byte[]", source);
        Assert.Contains("BindVertex", source);
        Assert.Contains("BindFragment", source);
        Assert.Contains("ResourceKind.Buffer", source);
        Assert.Contains("ResourceKind.Sampler", source);
    }

    [Fact]
    public void GeneratorAcceptsGpuStructGraphicsVaryingWithPosition()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;

            namespace Scratch;

            [GpuStruct]
            public partial struct VertexOut
            {
                [Position]
                public float4 Position;
                public float3 Normal;
            }

            [VertexShader]
            public readonly partial struct MeshVS : IVertexShader<VertexOut>
            {
                public VertexOut Execute()
                {
                    return default;
                }
            }

            [FragmentShader]
            public readonly partial struct MeshFS : IFragmentShader<VertexOut>
            {
                public float4 Execute(VertexOut input)
                {
                    return input.Position;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        Assert.Contains(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("VertexOut.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        Assert.Contains(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("MeshVS_MeshFS.Feather.Graphics.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsUnsupportedGraphicsVaryingShape()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;

            namespace Scratch;

            public struct BadVarying
            {
                public float4 Position;
            }

            [VertexShader]
            public readonly partial struct BadVS : IVertexShader<BadVarying>
            {
                public BadVarying Execute()
                {
                    return default;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0023");
    }

    [Fact]
    public void GeneratorReportsUnsupportedFragmentOutputShape()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;

            namespace Scratch;

            [FragmentShader]
            public readonly partial struct BadFS : IFragmentShader<float4, decimal>
            {
                public decimal Execute(float4 input)
                {
                    return 0;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0024");
    }

    [Theory]
    [InlineData("float3", "return new float3(1, 0, 0);")]
    [InlineData("BadNoColor", "return new BadNoColor { Target0 = new float4(1, 0, 0, 1) };")]
    [InlineData("BadDuplicateColor", "return new BadDuplicateColor { Target0 = new float4(1, 0, 0, 1), Target1 = new float4(0, 1, 0, 1) };")]
    [InlineData("BadGapColor", "return new BadGapColor { Target0 = new float4(1, 0, 0, 1), Target2 = new float4(0, 0, 1, 1) };")]
    [InlineData("BadNonFloat4Color", "return new BadNonFloat4Color { Target0 = new float3(1, 0, 0) };")]
    [InlineData("BadColorOverflow", "return new BadColorOverflow { Target8 = new float4(1, 0, 0, 1) };")]
    public void GeneratorReportsUnsupportedFragmentOutputShapesBeforeNativeLowering(string outputType, string returnExpression)
    {
        var compilation = CreateCompilation($$"""
            using Feather;
            using Feather.Math;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadNoColor
            {
                public float4 Target0;
            }

            [GpuStruct]
            public partial struct BadDuplicateColor
            {
                [Color(0)] public float4 Target0;
                [Color(0)] public float4 Target1;
            }

            [GpuStruct]
            public partial struct BadGapColor
            {
                [Color(0)] public float4 Target0;
                [Color(2)] public float4 Target2;
            }

            [GpuStruct]
            public partial struct BadNonFloat4Color
            {
                [Color(0)] public float3 Target0;
            }

            [GpuStruct]
            public partial struct BadColorOverflow
            {
                [Color(8)] public float4 Target8;
            }

            [FragmentShader]
            public readonly partial struct BadFS : IFragmentShader<float4, {{outputType}}>
            {
                public {{outputType}} Execute(float4 input)
                {
                    {{returnExpression}}
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0024");
    }

    [Fact]
    public void GeneratorEmitsUniformPushConstantMetadataAndPackers()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct ScaleKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output, Uniform<float4> scale) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = input[i];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("ScaleKernel.Feather.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();

        Assert.Contains("PushConstantDescriptor[]", source);
        Assert.Contains("typeof(global::Feather.Math.float4)", source);
        Assert.Contains("scale.Value", source);
        Assert.Contains("GpuValueLayout<global::Feather.Math.float4>.FieldSizeInBytes", source);
        Assert.Contains("GpuValueLayout<global::Feather.Math.float4>.PackValue", source);
        Assert.Contains("command.SetPushConstants", source);

        var module = ReadGeneratedIr(source);

        Assert.Equal(1u, module.PushConstantCount);
        Assert.Contains(module.Resources, resource => resource.Name == "scale" && resource.Kind == FeatherIrResourceKind.PushConstant && resource.ElementType == "Feather.Math.float4");
    }

    [Fact]
    public void GeneratorAcceptsUnsignedUniformPushConstantMetadata()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct OffsetKernel(ReadOnlyBuffer<uint> input, ReadWriteBuffer<uint> output, Uniform<uint> offset) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = input[i] + offset.Value;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("OffsetKernel.Feather.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();

        Assert.Contains("typeof(uint)", source);
        Assert.Contains("offset.Value", source);
        Assert.Contains("GpuValueLayout<uint>.FieldSizeInBytes", source);
        Assert.Contains("GpuValueLayout<uint>.PackValue", source);

        var module = ReadGeneratedIr(source);

        Assert.Equal(1u, module.PushConstantCount);
        Assert.Contains(module.Resources, resource => resource.Name == "offset"
            && resource.Kind == FeatherIrResourceKind.PushConstant
            && resource.ElementType == "uint");
    }

    [Fact]
    public void GeneratorEmitsStructuredControlFlowMarkersInIr()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct FlowKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    if (i > 0)
                    {
                        output[i] = input[i];
                    }
                    else
                    {
                        output[i] = input[0];
                    }

                    for (int j = 0; j < 2; j++)
                    {
                        output[i] = input[i];
                    }
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("FlowKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());
        var opcodes = module.Instructions.Select(instruction => instruction.Opcode).ToArray();

        Assert.Contains(FeatherIrInstructionOpcode.If, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.For, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.Else, opcodes);
        Assert.Equal(3, opcodes.Count(opcode => opcode == FeatherIrInstructionOpcode.BeginBlock));
        Assert.Equal(3, opcodes.Count(opcode => opcode == FeatherIrInstructionOpcode.EndBlock));
        Assert.True(IndexOf(opcodes, FeatherIrInstructionOpcode.If) < IndexOf(opcodes, FeatherIrInstructionOpcode.BeginBlock));
        Assert.True(IndexOf(opcodes, FeatherIrInstructionOpcode.Else) < Array.LastIndexOf(opcodes, FeatherIrInstructionOpcode.EndBlock));
    }

    [Fact]
    public void GeneratorLowersElementwiseArithmeticAssignmentsFromSyntax()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct ArithmeticKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = input[i] * 2.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("ArithmeticKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var assignment = Assert.Single(module.ElementwiseAssignments);
        Assert.Equal(FeatherIrAssignmentOperation.Multiply, assignment.Operation);
        Assert.Equal(FeatherIrAssignmentOperandKind.Literal, assignment.RightOperandKind);
        Assert.Equal("2", assignment.RightLiteral);
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.Assignment
            && instruction.OperandKind == FeatherIrOperandKind.ElementwiseAssignment
            && instruction.Operand == "ASSIGN1|output|i|mul|input|2");
    }

    [Fact]
    public void GeneratorLowersCommutativeLiteralArithmeticFromSemanticModel()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct ArithmeticKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = 2.0f * input[i];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("ArithmeticKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.Assignment
            && instruction.OperandKind == FeatherIrOperandKind.ElementwiseAssignment
            && instruction.Operand == "ASSIGN1|output|i|mul|input|2");
    }

    [Fact]
    public void GeneratorEmitsTypedExpressionSectionForNestedElementwiseArithmetic()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct NestedArithmeticKernel(ReadOnlyBuffer<float> input, ReadOnlyBuffer<float> bias, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = (input[i] * 2.0f) + bias[i];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("NestedArithmeticKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Empty(module.ElementwiseAssignments);
        var assignment = Assert.Single(module.ElementwiseExpressionAssignments);
        Assert.Equal(2u, assignment.DestinationBinding);
        Assert.Equal("i", assignment.Index);
        Assert.Equal(FeatherIrExpressionNodeKind.Binary, assignment.Expression.Kind);
        Assert.Equal(FeatherIrExpressionOperation.Add, assignment.Expression.Operation);
        Assert.Equal(FeatherIrExpressionOperation.Multiply, assignment.Expression.Left!.Operation);
        Assert.Equal(0u, assignment.Expression.Left.Left!.ResourceBinding);
        Assert.Equal("2", assignment.Expression.Left.Right!.Literal);
        Assert.Equal(1u, assignment.Expression.Right!.ResourceBinding);
    }

    [Fact]
    public void GeneratorEmitsTypedExpressionInvocationNodeForShaderMathIntrinsic()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct IntrinsicExpressionKernel(ReadOnlyBuffer<float> input, ReadOnlyBuffer<float> bias, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = ShaderMath.Sqrt(input[i]) + bias[i];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("IntrinsicExpressionKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var assignment = Assert.Single(module.ElementwiseExpressionAssignments);
        Assert.Equal(FeatherIrExpressionNodeKind.Binary, assignment.Expression.Kind);
        Assert.Equal(FeatherIrExpressionOperation.Add, assignment.Expression.Operation);
        var invocation = assignment.Expression.Left!;
        Assert.Equal(FeatherIrExpressionNodeKind.Invocation, invocation.Kind);
        Assert.Equal("global::Feather.Math.ShaderMath.Sqrt", invocation.Symbol);
        var argument = Assert.Single(invocation.Arguments);
        Assert.Equal(FeatherIrExpressionNodeKind.Resource, argument.Kind);
        Assert.Equal(0u, argument.ResourceBinding);
        Assert.Equal("i", argument.Index);
        Assert.Equal(1u, assignment.Expression.Right!.ResourceBinding);
    }

    [Fact]
    public void GeneratorPreservesMultipleElementwiseAssignmentRecords()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
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
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("MultiAssignmentKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Equal(2, module.Instructions.Count(instruction => instruction.Opcode == FeatherIrInstructionOpcode.Assignment));
        Assert.Equal(2, module.ElementwiseExpressionAssignments.Count);
        Assert.Contains(module.ElementwiseExpressionAssignments, assignment => assignment.DestinationBinding == 2
            && assignment.Expression.Kind == FeatherIrExpressionNodeKind.Binary);
        Assert.Contains(module.ElementwiseExpressionAssignments, assignment => assignment.DestinationBinding == 3
            && assignment.Expression.Kind == FeatherIrExpressionNodeKind.Resource
            && assignment.Expression.ResourceBinding == 0);
        var copy = Assert.Single(module.ElementwiseAssignments);
        Assert.Equal(3u, copy.DestinationBinding);
        Assert.Equal(0u, copy.LeftBinding);
        Assert.Equal(FeatherIrAssignmentOperation.Copy, copy.Operation);
    }

    [Fact]
    public void GeneratorEmitsPushConstantExpressionNodeForUniformValue()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct UniformExpressionKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output, Uniform<float> scale) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = input[i] * scale.Value;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("UniformExpressionKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var assignment = Assert.Single(module.ElementwiseExpressionAssignments);
        Assert.Equal(FeatherIrExpressionNodeKind.Binary, assignment.Expression.Kind);
        Assert.Equal(FeatherIrExpressionOperation.Multiply, assignment.Expression.Operation);
        Assert.Equal(0u, assignment.Expression.Left!.ResourceBinding);
        Assert.Equal(FeatherIrExpressionNodeKind.PushConstant, assignment.Expression.Right!.Kind);
        Assert.Equal(2u, assignment.Expression.Right.ResourceBinding);
        Assert.Contains(module.Resources, resource => resource.Name == "scale"
            && resource.Kind == FeatherIrResourceKind.PushConstant
            && resource.ElementType == "float");
    }

    [Fact]
    public void GeneratorEmitsFloat2ExpressionTreeForVectorArithmetic()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
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
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("Float2ExpressionKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var assignment = Assert.Single(module.ElementwiseExpressionAssignments);
        Assert.Equal(FeatherIrExpressionNodeKind.Binary, assignment.Expression.Kind);
        Assert.Equal(FeatherIrExpressionOperation.Add, assignment.Expression.Operation);
        Assert.Equal("Feather.Math.float2", assignment.Expression.TypeName);
        Assert.Equal(FeatherIrExpressionNodeKind.Binary, assignment.Expression.Left!.Kind);
        Assert.Equal(FeatherIrExpressionOperation.Multiply, assignment.Expression.Left.Operation);
        Assert.Equal(FeatherIrExpressionNodeKind.PushConstant, assignment.Expression.Left.Right!.Kind);
        Assert.Equal("float", assignment.Expression.Left.Right.TypeName);
        Assert.Equal(FeatherIrExpressionNodeKind.Resource, assignment.Expression.Right!.Kind);
        Assert.Equal(1u, assignment.Expression.Right.ResourceBinding);
    }

    [Fact]
    public void GeneratorEmitsFloat4ExpressionTreeForVectorArithmetic()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct Float4ExpressionKernel(
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
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("Float4ExpressionKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var assignment = Assert.Single(module.ElementwiseExpressionAssignments);
        Assert.Equal(FeatherIrExpressionNodeKind.Binary, assignment.Expression.Kind);
        Assert.Equal(FeatherIrExpressionOperation.Add, assignment.Expression.Operation);
        Assert.Equal("Feather.Math.float4", assignment.Expression.TypeName);
        Assert.Equal(FeatherIrExpressionNodeKind.Resource, assignment.Expression.Left!.Kind);
        Assert.Equal(FeatherIrExpressionNodeKind.PushConstant, assignment.Expression.Right!.Kind);
        Assert.Equal("Feather.Math.float4", assignment.Expression.Right.TypeName);
    }

    [Fact]
    public void GeneratorEmitsDotInvocationExpressionNodeForVectorIntrinsic()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
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
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("DotExpressionKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var assignment = Assert.Single(module.ElementwiseExpressionAssignments);
        Assert.Equal(FeatherIrExpressionNodeKind.Invocation, assignment.Expression.Kind);
        Assert.Equal("global::Feather.Math.ShaderMath.Dot", assignment.Expression.Symbol);
        Assert.Equal("float", assignment.Expression.TypeName);
        Assert.Equal(2, assignment.Expression.Arguments.Count);
        Assert.All(assignment.Expression.Arguments, argument => Assert.Equal(FeatherIrExpressionNodeKind.Resource, argument.Kind));
    }

    [Theory]
    [InlineData("output")]
    [InlineData("input")]
    [InlineData("texture")]
    [InlineData("buffer")]
    [InlineData("uniform")]
    [InlineData("layout")]
    [InlineData("in")]
    [InlineData("out")]
    [InlineData("sampler")]
    [InlineData("image")]
    public void GeneratorAcceptsReservedGlslLocalIdentifiers(string localName)
    {
        var csharpLocalName = localName is "in" or "out" ? "@" + localName : localName;
        var compilation = CreateCompilation($$"""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct ReservedIdentifierKernel(ReadOnlyBuffer<float> source, ReadWriteBuffer<float> destination) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float {{csharpLocalName}} = source[i];
                    destination[i] = {{csharpLocalName}};
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        Assert.Contains(outputCompilation.SyntaxTrees, tree =>
            tree.FilePath.EndsWith("ReservedIdentifierKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorAcceptsGptStyleComplexIntegerBufferIndexExpressions()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            internal static class KernelConstants
            {
                public const int EmbedDim = 8;
                public const int Seq = 4;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct GptIndexKernel(
                ReadOnlyBuffer<int> tokens,
                ReadOnlyBuffer<float> tokenEmbedding,
                ReadOnlyBuffer<float> attentionWeights,
                ReadWriteBuffer<float> scratch) : IKernel1D
            {
                public void Execute()
                {
                    int batch = ThreadIds.X;
                    for (int pos = 0; pos < KernelConstants.Seq; pos++)
                    {
                        int token = tokens[(batch * KernelConstants.Seq) + pos];
                        for (int o = 0; o < KernelConstants.EmbedDim; o++)
                        {
                            float sum = 0.0f;
                            for (var i = 0; i < KernelConstants.EmbedDim; i++)
                            {
                                sum += tokenEmbedding[(token * KernelConstants.EmbedDim) + i]
                                    * attentionWeights[(1 * KernelConstants.EmbedDim * KernelConstants.EmbedDim) + (o * KernelConstants.EmbedDim) + i];
                            }

                            int kBase = batch * KernelConstants.Seq * KernelConstants.EmbedDim;
                            scratch[kBase + (pos * KernelConstants.EmbedDim) + o] = sum;
                        }
                    }
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        Assert.Contains(outputCompilation.SyntaxTrees, tree =>
            tree.FilePath.EndsWith("GptIndexKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorResolvesVarAssignedFromIntegerExpressionsInTypedIr()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct VarIntegerKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    var i = ThreadIds.X;
                    var baseIndex = (i * 4) + 1;
                    var offset = baseIndex % 3;
                    output[baseIndex + offset] = input[baseIndex - offset];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = GetGeneratedTree(outputCompilation, diagnostics, "VarIntegerKernel.Feather.g.cs");
        var section = ReadTypedIrSection(ExtractTypedIrBytes(generated.ToString()));

        Assert.DoesNotContain("var", section.Strings);
        Assert.Contains(section.Expressions, expression => expression.Kind == 7 && expression.Op == (uint)ShaderBinaryOperator.Modulo);
    }

    [Fact]
    public void GeneratorAcceptsKernelLocalAndStaticConstantsInIndexExpressions()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            internal static class KernelConstants
            {
                public const int StaticStride = 8;
                public static readonly int ReadonlyOffset = 2;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct ConstantIndexKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    const int LocalScale = 2;
                    int i = ThreadIds.X;
                    output[(i * KernelConstants.StaticStride) + KernelConstants.ReadonlyOffset] =
                        input[(i * LocalScale) + KernelConstants.ReadonlyOffset];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = GetGeneratedTree(outputCompilation, diagnostics, "ConstantIndexKernel.Feather.g.cs");
        var section = ReadTypedIrSection(ExtractTypedIrBytes(generated.ToString()));

        Assert.Contains("8", section.Strings);
        Assert.Contains("2", section.Strings);
        Assert.Contains(section.Expressions, expression => expression.Kind == 1
            && expression.NameId < section.Strings.Count
            && section.Strings[(int)expression.NameId] == "2");
    }

    [Fact]
    public void AnalyzerReportsTopLevelConstantsReferencedFromKernel()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            const int Stride = 4;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadTopLevelConstantKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i * Stride] = input[i];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0028");
        Assert.Contains("Stride", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.Contains("static const or static readonly", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void GeneratorEmitsCrossInvocationExpressionNodeForVectorIntrinsic()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
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
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("CrossExpressionKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var assignment = Assert.Single(module.ElementwiseExpressionAssignments);
        Assert.Equal(FeatherIrExpressionNodeKind.Invocation, assignment.Expression.Kind);
        Assert.Equal("global::Feather.Math.ShaderMath.Cross", assignment.Expression.Symbol);
        Assert.Equal("Feather.Math.float3", assignment.Expression.TypeName);
        Assert.Equal(2, assignment.Expression.Arguments.Count);
        Assert.All(assignment.Expression.Arguments, argument => Assert.Equal(FeatherIrExpressionNodeKind.Resource, argument.Kind));
    }

    [Fact]
    public void GeneratorEmitsStructuredTexture2DCopyAssignment()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1, 1)]
            public readonly partial struct TextureCopyKernel(ReadOnlyTexture2D<float4> input, ReadWriteTexture2D<float4> output) : IKernel2D
            {
                public void Execute()
                {
                    int2 p = ThreadIds.XY;
                    output[p] = input[p];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("TextureCopyKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Contains(module.Resources, resource => resource.Name == "input"
            && resource.Kind == FeatherIrResourceKind.Texture2D
            && resource.Access == FeatherIrResourceAccess.Read);
        Assert.Contains(module.Resources, resource => resource.Name == "output"
            && resource.Kind == FeatherIrResourceKind.Texture2D
            && resource.Access == FeatherIrResourceAccess.ReadWrite);
        var assignment = Assert.Single(module.ElementwiseAssignments);
        Assert.Equal(1u, assignment.DestinationBinding);
        Assert.Equal(0u, assignment.LeftBinding);
        Assert.Equal(FeatherIrAssignmentOperation.Copy, assignment.Operation);
        Assert.Equal("p", assignment.Index);
    }

    [Fact]
    public void SemanticLowererKeepsElementwiseAssignmentAsTypedModel()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct ArithmeticKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = 2.0f * input[i];
                }
            }
            """);
        var syntaxTree = compilation.SyntaxTrees.Single();
        var syntax = syntaxTree.GetRoot().DescendantNodes().OfType<StructDeclarationSyntax>().Single();
        var semanticModel = compilation.GetSemanticModel(syntaxTree);
        var symbol = semanticModel.GetDeclaredSymbol(syntax)!;
        var created = ShaderModelFactory.Create(syntax, symbol)!;

        var model = created with
        {
            LoweredInstructions = ShaderSemanticLowerer.Lower(
                created,
                semanticModel,
                CancellationToken.None)
        };

        var lowered = Assert.Single(model.LoweredInstructions.Items, instruction => instruction.Kind == LoweredShaderInstructionKind.ElementwiseAssignment);
        var assignment = Assert.IsType<LoweredElementwiseAssignmentModel>(lowered.ElementwiseAssignment);
        Assert.Equal("output", assignment.DestinationResourceName);
        Assert.Equal("i", assignment.IndexName);
        Assert.Equal(LoweredElementwiseAssignmentOperation.Multiply, assignment.Operation);
        Assert.Equal("input", assignment.LeftOperand);
        Assert.Equal(LoweredElementwiseAssignmentOperandKind.Literal, assignment.RightOperandKind);
        Assert.Equal("2", assignment.RightOperand);
    }

    [Fact]
    public void GeneratorEmitsTexture3DResourceMetadata()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct VolumeKernel(ReadWriteTexture3D<float4> volume) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("VolumeKernel.Feather.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();
        var module = ReadGeneratedIr(source);

        Assert.Contains("ResourceKind.Texture3D", source);
        Assert.Contains(module.Resources, resource => resource.Name == "volume" && resource.Kind == FeatherIrResourceKind.Texture3D && resource.Access == FeatherIrResourceAccess.ReadWrite);
    }

    [Fact]
    public void GeneratorEmitsTexture3DShaderIndexingInTypedIr()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1, 1, 1)]
            public readonly partial struct VolumeKernel(ReadWriteTexture3D<float4> volume) : IKernel3D
            {
                public void Execute()
                {
                    volume[ThreadIds.XYZ] = new float4(1, 0, 0, 1);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree =>
            tree.FilePath.EndsWith("VolumeKernel.Feather.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();
        var module = ReadGeneratedIr(source);

        Assert.Contains("ResourceKind.Texture3D", source);
        Assert.Contains(module.Resources, resource =>
            resource.Name == "volume" &&
            resource.Kind == FeatherIrResourceKind.Texture3D &&
            resource.Access == FeatherIrResourceAccess.ReadWrite);
    }

    [Fact]
    public void GeneratorEmitsBarrierInstructionsInIr()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BarrierKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    GpuBarrier.Workgroup();
                    GpuBarrier.Memory();
                    GpuBarrier.Full();
                    output[i] = input[i];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("BarrierKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.WorkgroupBarrier
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "global::Feather.GpuBarrier.Workgroup");
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.MemoryBarrier
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "global::Feather.GpuBarrier.Memory");
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.FullBarrier
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "global::Feather.GpuBarrier.Full");
    }

    [Fact]
    public void GeneratorLowersIntrinsicCallsFromSemanticSymbols()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;
            using static Feather.GpuBarrier;
            using static Feather.GpuAtomic;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct SemanticIntrinsicKernel(ReadWriteBuffer<int> output) : IKernel1D
            {
                public void Execute()
                {
                    Workgroup();
                    _ = Add(ref output[0], ThreadIds.X);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = GetGeneratedTree(outputCompilation, diagnostics, "SemanticIntrinsicKernel.Feather.g.cs");
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.WorkgroupBarrier
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "global::Feather.GpuBarrier.Workgroup");
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.AtomicAdd
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "global::Feather.GpuAtomic.Add");
    }

    [Fact]
    public void GeneratorPreservesKnownMathSymbolsAndAdMetadata()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.AD;
            using Feather.Math;
            using Feather.Resources;
            using static Feather.Math.Hlsl;

            namespace Scratch;

            [Kernel]
            [AutoDiff]
            [ThreadGroupSize(1)]
            public readonly partial struct MathSymbolKernel(ReadWriteBuffer<float> parameters, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float value = parameters[i] + Dot(new float3(1, 0, 0), new float3(0, 1, 0));
                    value = ShaderMath.Sin(value);
                    AD.Parameter(parameters[i]);
                    AD.Loss(value);
                    output[i] = value;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("MathSymbolKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.Invocation
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "global::Feather.Math.Hlsl.Dot");
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.Invocation
            && instruction.OperandKind == FeatherIrOperandKind.Symbol
            && instruction.Operand == "global::Feather.Math.ShaderMath.Sin");
        Assert.Contains(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Parameter
            && annotation.Name == "parameters"
            && annotation.Binding == 0u);
        Assert.Contains(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Loss
            && annotation.Name == "value");
        Assert.DoesNotContain(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.Invocation
            && instruction.Operand.Contains("Feather.AD.AD", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorEmitsAtomicInstructionsInIr()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct AtomicKernel(ReadWriteBuffer<int> output) : IKernel1D
            {
                public void Execute()
                {
                    _ = GpuAtomic.Add(ref output[0], 1);
                    _ = GpuAtomic.Sub(ref output[1], 1);
                    _ = GpuAtomic.Min(ref output[2], 0);
                    _ = GpuAtomic.Max(ref output[3], 10);
                    _ = GpuAtomic.And(ref output[4], 0xFF);
                    _ = GpuAtomic.Or(ref output[5], 0x100);
                    _ = GpuAtomic.Xor(ref output[6], 0x1);
                    _ = GpuAtomic.Exchange(ref output[7], 42);
                    _ = GpuAtomic.CompareExchange(ref output[8], 42, 100);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("AtomicKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());
        var opcodes = module.Instructions.Select(instruction => instruction.Opcode).ToHashSet();

        Assert.Contains(FeatherIrInstructionOpcode.AtomicAdd, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.AtomicSub, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.AtomicMin, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.AtomicMax, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.AtomicAnd, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.AtomicOr, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.AtomicXor, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.AtomicExchange, opcodes);
        Assert.Contains(FeatherIrInstructionOpcode.AtomicCompareExchange, opcodes);
    }

    [Fact]
    public void GeneratorEmitsSharedMemoryDeclarationInIr()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct SharedKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    var shared = new SharedMemory<float>(256);
                    int i = ThreadIds.X;
                    shared[i] = 1;
                    GpuBarrier.Workgroup();
                    output[i] = shared[i];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("SharedKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.SharedMemoryDeclaration && instruction.Operand.Contains("SharedMemory<float>", StringComparison.Ordinal));
        Assert.Contains(module.Instructions, instruction => instruction.Opcode == FeatherIrInstructionOpcode.WorkgroupBarrier);
    }

    [Fact]
    public void GeneratorEmitsGpuStructLayoutContract()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;

            namespace Scratch;

            [GpuStruct]
            public partial struct Scene
            {
                public float3 LightDir;
                public float Intensity;
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("Scene.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();

        Assert.Contains("IGpuStruct<global::Scratch.Scene>", source);
        Assert.Contains("new(\"LightDir\", typeof(global::Feather.Math.float3), 0, 12, 16)", source);
        Assert.Contains("new(\"Intensity\", typeof(float), 12, 4, 4)", source);
        Assert.Contains("__feather_destination.Clear()", source);
        Assert.Contains("GpuValueLayout<global::Feather.Math.float3>.PackValue", source);
        Assert.Contains("GpuValueLayout<global::Feather.Math.float3>.UnpackValue", source);
    }

    [Fact]
    public void GeneratorEmitsRecordGpuStructLayoutContract()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("Rgba32.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();

        Assert.Contains("partial record struct Rgba32", source);
        Assert.Contains("new(\"R\", typeof(byte), 0, 1, 1)", source);
        Assert.Contains("new(\"G\", typeof(byte), 1, 1, 1)", source);
        Assert.Contains("new(\"B\", typeof(byte), 2, 1, 1)", source);
        Assert.Contains("new(\"A\", typeof(byte), 3, 1, 1)", source);
        Assert.Contains("new global::Scratch.Rgba32(__feather_field_R, __feather_field_G, __feather_field_B, __feather_field_A)", source);
    }

    [Fact]
    public void GeneratorEmitsUintRecordGpuStructLayoutContract()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public readonly partial record struct Pixel(uint R, uint G, uint B, uint A);
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("Pixel.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();

        Assert.Contains("partial record struct Pixel", source);
        Assert.Contains("new(\"R\", typeof(uint), 0, 4, 4)", source);
        Assert.Contains("new global::Scratch.Pixel(__feather_field_R, __feather_field_G, __feather_field_B, __feather_field_A)", source);
    }

    [Fact]
    public void GeneratorReportsComputedGpuStructPropertyAndSuppressesSource()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadLayout
            {
                public float Value;
                public float Twice => Value * 2.0f;
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0019");
        Assert.Contains("property 'Twice' is not a record primary-constructor storage property", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsReadonlyGpuStructFieldAndSuppressesSource()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float Value;
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0019");
        Assert.Contains("field 'Value' is readonly and cannot be written by generated unpack", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorRejectsReadonlyGpuStructAsBufferElementAndSuppressesSources()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float3 Value;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadOnlyBuffer<BadLayout> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = 0.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0019"
            && diagnostic.GetMessage().Contains("field 'Value' is readonly", StringComparison.Ordinal));
        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0004"
            && diagnostic.GetMessage().Contains("BadLayout", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorRejectsNestedInvalidGpuStructAsBufferElementAndSuppressesSources()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct Inner
            {
                public readonly float Value;
            }

            [GpuStruct]
            public partial struct Outer
            {
                public Inner Inner;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadOnlyBuffer<Outer> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = 0.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0019"
            && diagnostic.GetMessage().Contains("field 'Value' is readonly", StringComparison.Ordinal));
        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0004"
            && diagnostic.GetMessage().Contains("Outer", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("Inner.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("Outer.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsNonPartialGpuStructAndSuppressesSource()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public struct BadLayout
            {
                public float Value;
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0019");
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerReportsNonReadonlyKernel()
    {
        var compilation = CreateCompilation("""
            using Feather;

            [Kernel]
            public partial struct BadKernel : IKernel1D
            {
                public void Execute() { }
            }
            """);

        var analyzer = new FeatherAnalyzer();
        var diagnostics = await compilation.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(analyzer)).GetAnalyzerDiagnosticsAsync();

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0001");
    }

    [Theory]
    [InlineData("""
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            public void Run() { }
        }
        """)]
    [InlineData("""
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            [Entry]
            public void Run() { }

            [Entry]
            public void Main() { }
        }
        """)]
    [InlineData("""
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            [Entry]
            public float Run() => 0.0f;
        }
        """)]
    [InlineData("""
        using Feather;
        using Feather.Math;

        [VertexShader]
        public readonly partial struct BadVertex : IVertexShader<float4>
        {
            [Entry]
            public float3 Run() => new(0.0f);
        }
        """)]
    [InlineData("""
        using Feather;
        using Feather.Math;

        [FragmentShader]
        public readonly partial struct BadFragment : IFragmentShader<float4>
        {
            [Entry]
            public float4 Run(float3 input) => new(0.0f);
        }
        """)]
    public async Task AnalyzerReportsInvalidEntryPoint(string source)
    {
        var diagnostics = await AnalyzeAsync(source);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0003");
    }

    [Theory]
    [InlineData("FE0005", """
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            private readonly string captured;
            public void Execute() { }
        }
        """)]
    [InlineData("FE0009", """
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            public void Execute()
            {
                switch (ThreadIds.X)
                {
                    case 0: return;
                }
            }
        }
        """)]
    [InlineData("FE0010", """
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            public void Execute()
            {
                Helper<int>();
            }

            [ShaderFunction]
            private static void Helper<T>() { }
        }
        """)]
    [InlineData("FE0014", """
        using Feather;

        public interface IThing
        {
            void Run();
        }

        [Kernel]
        public readonly partial struct BadKernel(IThing thing) : IKernel1D
        {
            public void Execute()
            {
                thing.Run();
            }
        }
        """)]
    [InlineData("FE0015", """
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            public void Execute()
            {
                Recurse();
            }

            [ShaderFunction]
            private static void Recurse()
            {
                Recurse();
            }
        }
        """)]
    [InlineData("FE0016", """
        using Feather;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(ReadOnlyBuffer<float> input) : IKernel1D
        {
            public void Execute()
            {
                input[0] = 1;
            }
        }
        """)]
    [InlineData("FE0016", """
        using Feather;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(WriteOnlyBuffer<float> output) : IKernel1D
        {
            public void Execute()
            {
                output[0] += 1;
            }
        }
        """)]
    [InlineData("FE0016", """
        using Feather;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(ReadOnlyBuffer<float> input) : IKernel1D
        {
            public void Execute()
            {
                input[0] += 1;
            }
        }
        """)]
    [InlineData("FE0016", """
        using Feather;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(WriteOnlyBuffer<float> output) : IKernel1D
        {
            public void Execute()
            {
                float value = output[0];
            }
        }
        """)]
    [InlineData("FE0017", """
        using Feather;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(ReadOnlyBuffer<float> input) : IKernel1D
        {
            public void Execute()
            {
                float value = input[1.5f];
            }
        }
        """)]
    [InlineData("FE0018", """
        using Feather;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(ReadOnlyTexture2D<float> input) : IKernel1D
        {
            public void Execute()
            {
                float value = input[0];
            }
        }
        """)]
    [InlineData("FE0018", """
        using Feather;
        using Feather.Math;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(ReadOnlyTexture3D<float> input) : IKernel3D
        {
            public void Execute()
            {
                float value = input[ThreadIds.XY];
            }
        }
        """)]
    [InlineData("FE0016", """
        using Feather;
        using Feather.Math;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(WriteOnlyTexture2D<float4> output) : IKernel2D
        {
            public void Execute()
            {
                float4 value = output[ThreadIds.XY];
            }
        }
        """)]
    [InlineData("FE0016", """
        using Feather;
        using Feather.Math;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(ReadOnlyTexture2D<float4> input) : IKernel2D
        {
            public void Execute()
            {
                input[ThreadIds.XY] = new float4(1, 2, 3, 4);
            }
        }
        """)]
    [InlineData("FE0008", """
        using Feather;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(SamplerState sampler) : IKernel1D
        {
            public void Execute()
            {
                sampler.Dispose();
            }
        }
        """)]
    [InlineData("FE0008", """
        using Feather;
        using Feather.Resources;

        public static class Helpers
        {
            [Callable]
            public static float External(float value) => value;
        }

        [Kernel]
        public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
        {
            public void Execute()
            {
                output[ThreadIds.X] = Helpers.External(1);
            }
        }
        """)]
    [InlineData("FE0008", """
        using Feather;
        using Feather.Resources;

        [Kernel]
        public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
        {
            public void Execute()
            {
                float value = 1;
                Mutate(ref value);
                output[ThreadIds.X] = value;
            }

            [Callable]
            private static void Mutate(ref float value)
            {
                value = value + 1;
            }
        }
        """)]
    [InlineData("FE0011", """
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            public void Execute()
            {
                int[] values = new int[4];
            }
        }
        """)]
    [InlineData("FE0007", """
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            public void Execute()
            {
                string text = "shader";
            }
        }
        """)]
    [InlineData("FE0007", """
        using Feather;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            public unsafe void Execute()
            {
                int* values = stackalloc int[4];
            }
        }
        """)]
    [InlineData("FE0013", """
        using Feather;
        using System.Threading.Tasks;

        [Kernel]
        public readonly partial struct BadKernel : IKernel1D
        {
            public async void Execute()
            {
            }
        }
        """)]
    public async Task AnalyzerReportsAdditionalUnsupportedShaderDiagnostics(string id, string source)
    {
        var diagnostics = await AnalyzeAsync(source);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == id);
    }

    [Fact]
    public async Task AnalyzerReportsUnsupportedConstructInsideCallableBody()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = SelectValue(ThreadIds.X);
                }

                [Callable]
                private static float SelectValue(int value)
                {
                    switch (value)
                    {
                        case 0:
                            return 1;
                        default:
                            return 2;
                    }
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0009");
    }

    [Fact]
    public async Task AnalyzerReportsMutuallyRecursiveCallables()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = A(1);
                }

                [Callable]
                private static float A(float value)
                {
                    return B(value);
                }

                [Callable]
                private static float B(float value)
                {
                    return A(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0015");
    }

    [Fact]
    public async Task AnalyzerReportsDirectlyRecursiveCallable()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = Self(1);
                }

                [Callable]
                private static float Self(float value)
                {
                    return value <= 0 ? 0 : Self(value - 1);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0015");
    }

    [Fact]
    public async Task AnalyzerReportsUnsupportedCallableParameterDeclaration()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = UsesString("bad");
                }

                [Callable]
                private static float UsesString(string value)
                {
                    return 1;
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0008");
    }

    [Fact]
    public void GeneratorReportsUnsupportedByteBackedGpuStructBufferElement()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadOnlyBuffer<Rgba32> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = 0.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0004");
        Assert.Contains("Rgba32", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsUnsupportedInvalidGpuStructUniform()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float Value;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(Uniform<BadLayout> value, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = 0.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0004"
            && diagnostic.GetMessage().Contains("BadLayout", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsUnsupportedByteBackedGpuStructUniform()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output, Uniform<Rgba32> color) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = 0.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0004");
        Assert.Contains("Rgba32", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsUnsupportedByteBackedGpuStructLocal()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel : IKernel1D
            {
                public void Execute()
                {
                    Rgba32 color = default;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0007");
        Assert.Contains("unsupported shader type", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.Contains("Rgba32", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsUnsupportedInvalidGpuStructLocal()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float Value;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel : IKernel1D
            {
                public void Execute()
                {
                    BadLayout value = default;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0007"
            && diagnostic.GetMessage().Contains("BadLayout", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsUnsupportedByteBackedGpuStructSharedMemory()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel : IKernel1D
            {
                public void Execute()
                {
                    var shared = new SharedMemory<Rgba32>(4);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0007");
        Assert.Contains("unsupported shader type", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.Contains("SharedMemory", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsUnsupportedInvalidGpuStructSharedMemory()
    {
        var compilation = CreateCompilation("""
            using Feather;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float Value;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel : IKernel1D
            {
                public void Execute()
                {
                    var shared = new SharedMemory<BadLayout>(4);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0007"
            && diagnostic.GetMessage().Contains("SharedMemory", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerReportsUnsupportedByteBackedGpuStructCallableParameter()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [GpuStruct]
            public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = Use(default);
                }

                [Callable]
                private static float Use(Rgba32 color) => 0.0f;
            }
            """);

        Assert.Single(diagnostics.Where(diagnostic => diagnostic.Id == "FE0008"));
    }

    [Fact]
    public async Task AnalyzerReportsUnsupportedInvalidGpuStructCallableParameter()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float Value;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = Use(default);
                }

                [Callable]
                private static float Use(BadLayout value) => 0.0f;
            }
            """);

        Assert.Single(diagnostics.Where(diagnostic => diagnostic.Id == "FE0008"));
    }

    [Fact]
    public async Task AnalyzerReportsUnsupportedByteBackedGpuStructCallableReturn()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [GpuStruct]
            public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    _ = Make();
                    output[ThreadIds.X] = 0.0f;
                }

                [Callable]
                private static Rgba32 Make() => default;
            }
            """);

        Assert.Single(diagnostics.Where(diagnostic => diagnostic.Id == "FE0008"));
    }

    [Fact]
    public async Task AnalyzerReportsUnsupportedInvalidGpuStructCallableReturn()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float Value;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    _ = Make();
                    output[ThreadIds.X] = 0.0f;
                }

                [Callable]
                private static BadLayout Make() => default;
            }
            """);

        Assert.Single(diagnostics.Where(diagnostic => diagnostic.Id == "FE0008"));
    }

    [Theory]
    [InlineData("ReadOnlyTexture2D<BadLayout>", "IKernel2D", "[ThreadGroupSize(1, 1)]")]
    [InlineData("WriteOnlyTexture2D<BadLayout>", "IKernel2D", "[ThreadGroupSize(1, 1)]")]
    [InlineData("ReadWriteTexture2D<BadLayout>", "IKernel2D", "[ThreadGroupSize(1, 1)]")]
    [InlineData("ReadWriteNormalizedTexture2D<BadLayout>", "IKernel2D", "[ThreadGroupSize(1, 1)]")]
    [InlineData("SampledTexture2D<BadLayout>", "IKernel2D", "[ThreadGroupSize(1, 1)]")]
    [InlineData("ReadOnlyTexture3D<BadLayout>", "IKernel3D", "[ThreadGroupSize(1, 1, 1)]")]
    [InlineData("WriteOnlyTexture3D<BadLayout>", "IKernel3D", "[ThreadGroupSize(1, 1, 1)]")]
    [InlineData("ReadWriteTexture3D<BadLayout>", "IKernel3D", "[ThreadGroupSize(1, 1, 1)]")]
    [InlineData("ReadWriteNormalizedTexture3D<BadLayout>", "IKernel3D", "[ThreadGroupSize(1, 1, 1)]")]
    public void GeneratorRejectsUnusedInvalidGpuStructTextureElementAndSuppressesSources(
        string textureType,
        string kernelInterface,
        string threadGroupAttribute)
    {
        var compilation = CreateCompilation($$"""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float Value;
            }

            [Kernel]
            {{threadGroupAttribute}}
            public readonly partial struct BadKernel({{textureType}} input, ReadWriteBuffer<float> output) : {{kernelInterface}}
            {
                public void Execute()
                {
                    output[0] = 0.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0004"
            && diagnostic.GetMessage().Contains("BadLayout", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorRejectsUnusedNestedInvalidGpuStructTextureElementAndSuppressesSources()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct Inner
            {
                public readonly float Value;
            }

            [GpuStruct]
            public partial struct Outer
            {
                public Inner Inner;
            }

            [Kernel]
            [ThreadGroupSize(1, 1)]
            public readonly partial struct BadKernel(ReadOnlyTexture2D<Outer> input, ReadWriteBuffer<float> output) : IKernel2D
            {
                public void Execute()
                {
                    output[0] = 0.0f;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0004"
            && diagnostic.GetMessage().Contains("Outer", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("Inner.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("Outer.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorAllowsByteBackedGpuStructTextureValue()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

            [Kernel]
            [ThreadGroupSize(1, 1, 1)]
            public readonly partial struct TextureKernel(ReadOnlyTexture2D<Rgba32> input, ReadWriteTexture2D<Rgba32> output) : IKernel2D
            {
                public void Execute()
                {
                    output[ThreadIds.XY] = input[ThreadIds.XY];
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        Assert.Contains(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("TextureKernel.Feather.g.cs", StringComparison.Ordinal));
        Assert.Contains(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("Rgba32.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorReportsBodyDiagnosticsAndSuppressesKernelSource()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadOnlyBuffer<float> input) : IKernel1D
            {
                public void Execute()
                {
                    input[0] = 1;
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0016");
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerReportsUnsupportedBodyDiagnosticOnce()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel : IKernel1D
            {
                public void Execute()
                {
                    string text = "shader";
                }
            }
            """);

        Assert.Single(diagnostics.Where(diagnostic => diagnostic.Id == "FE0007"));
    }

    [Fact]
    public void GeneratorReportsCallableBodyDiagnosticsAndSuppressesKernelSource()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = A(1);
                }

                [Callable]
                private static float A(float value)
                {
                    return B(value);
                }

                [Callable]
                private static float B(float value)
                {
                    return A(value);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0015");
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadKernel.Feather.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerAcceptsWriteOnlyBufferAsAssignmentTarget()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct GoodKernel(WriteOnlyBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = 1.0f;
                }
            }
            """);

        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Id == "FE0016");
    }

    [Fact]
    public async Task AnalyzerAcceptsSupportedScalarIntrinsicInElementwiseAssignment()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct GoodKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = ShaderMath.Sqrt(input[i]);
                }
            }
            """);

        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Id == "FE0026");
    }

    [Fact]
    public async Task AnalyzerAcceptsDotIntrinsicInsideElementwiseAssignment()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct GoodKernel(ReadOnlyBuffer<float3> left, ReadOnlyBuffer<float3> right, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = ShaderMath.Dot(left[i], right[i]);
                }
            }
            """);

        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Id == "FE0026");
    }

    [Fact]
    public async Task AnalyzerAcceptsCrossIntrinsicInsideElementwiseAssignment()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct GoodKernel(ReadOnlyBuffer<float3> left, ReadOnlyBuffer<float3> right, ReadWriteBuffer<float3> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = ShaderMath.Cross(left[i], right[i]);
                }
            }
            """);

        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Id == "FE0026");
    }

    [Fact]
    public async Task AnalyzerAcceptsNormalizeIntrinsicInsideElementwiseAssignment()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct GoodKernel(ReadOnlyBuffer<float3> input, ReadWriteBuffer<float3> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = ShaderMath.Normalize(input[i]);
                }
            }
            """);

        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Id == "FE0026");
    }

    [Fact]
    public async Task AnalyzerAcceptsVectorMathIntrinsicOverloadsInsideElementwiseAssignment()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct GoodKernel(ReadOnlyBuffer<float3> left, ReadOnlyBuffer<float3> right, ReadWriteBuffer<float3> output) : IKernel1D
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
            """);

        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Id == "FE0026");
    }

    [Fact]
    public async Task AnalyzerAcceptsMatrixMathIntrinsicOverloadsInsideElementwiseAssignment()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct GoodKernel(ReadWriteBuffer<float4> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float4x4 a = new float4x4(
                        new float4(1, 0, 0, 0),
                        new float4(0, 2, 0, 0),
                        new float4(0, 0, 4, 0),
                        new float4(0, 0, 0, 1));
                    float4x4 b = ShaderMath.Inverse(a);
                    float determinant = ShaderMath.Determinant(a);
                    float4x4 mixed = ShaderMath.Hadamard(ShaderMath.Transpose(a), b);
                    output[i] = Hlsl.Mul(ShaderMath.Mul(mixed, a), new float4(determinant, 1, 1, 1));
                }
            }
            """);

        Assert.DoesNotContain(diagnostics, diagnostic => diagnostic.Id == "FE0026");
    }

    [Fact]
    public async Task AnalyzerReportsUnsupportedIntrinsicInsideElementwiseAssignment()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadOnlyBuffer<float3> left, ReadOnlyBuffer<float3> right, ReadWriteBuffer<float3> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = new float3(ShaderMath.Tanh(left[i].X));
                }
            }
            """);

        var diagnostic = Assert.Single(diagnostics, diagnostic => diagnostic.Id == "FE0026");
        Assert.Contains("global::Feather.Math.ShaderMath.Tanh", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void GeneratorReportsUnsupportedGpuStructMatrixLayout()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using System.Numerics;

            [GpuStruct]
            public partial struct BadLayout
            {
                public Matrix4x4 Transform;
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0020");
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorEmitsGpuStructArrayLayoutContract()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Math;

            [GpuStruct]
            public partial struct ArrayScene
            {
                public GpuArray4<float3> Directions;
                public float Weight;
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("ArrayScene.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();

        Assert.Contains("new(\"Directions\", typeof(global::Feather.GpuArray4<global::Feather.Math.float3>), 0, 64, 16, 4, 16)", source);
        Assert.Contains("new(\"Weight\", typeof(float), 64, 4, 4)", source);
        Assert.Contains("GpuValueLayout<global::Feather.Math.float3>.PackValue", source);
        Assert.Contains("GpuValueLayout<global::Feather.Math.float3>.UnpackValue", source);
        Assert.Contains("__feather_array_index < 4", source);
    }

    [Fact]
    public void GeneratorEmitsNestedGpuStructArrayLayoutContract()
    {
        var compilation = CreateCompilation("""
        using Feather;

        [GpuStruct]
        public partial struct Inner
        {
            public float Value;
        }

        [GpuStruct]
        public partial struct ArrayScene
        {
            public GpuArray3<Inner> Items;
            public float Weight;
        }
        """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("ArrayScene.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
        var source = generated.ToString();

        Assert.Contains("new(\"Items\", typeof(global::Feather.GpuArray3<global::Inner>), 0, 12, 4, 3, 4)", source);
        Assert.Contains("new(\"Weight\", typeof(float), 12, 4, 4)", source);
        Assert.Contains("GpuValueLayout<global::Inner>.PackValue", source);
    }

    [Theory]
    [InlineData("Managed scalar array", """
        using Feather;

        [GpuStruct]
        public partial struct BadLayout
        {
            public int[] Values;
        }
        """, "Values")]
    [InlineData("Nested managed struct array", """
        using Feather;

        [GpuStruct]
        public partial struct Inner
        {
            public float Value;
        }

        [GpuStruct]
        public partial struct BadLayout
        {
            public Inner[] Items;
        }
        """, "Items")]
    [InlineData("Nested GpuArray wrapper", """
        using Feather;

        [GpuStruct]
        public partial struct BadLayout
        {
            public GpuArray2<GpuArray2<int>> Values;
        }
        """, "Values")]
    public void GeneratorReportsUnsupportedGpuStructArrayLayouts(string caseName, string source, string fieldName)
    {
        _ = caseName;
        var compilation = CreateCompilation(source);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        var diagnostic = Assert.Single(
            diagnostics,
            diagnostic => diagnostic.Id == "FE0019"
                && diagnostic.GetMessage().Contains($"field '{fieldName}' uses an unsupported GPU type", StringComparison.Ordinal));
        Assert.Contains($"field '{fieldName}' uses an unsupported GPU type", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(outputCompilation.SyntaxTrees, tree => tree.FilePath.EndsWith("BadLayout.Feather.GpuStruct.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void TypedIrLowererPreservesAllLocalDeclaratorsInSourceOrder()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct MultiLocalKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X, j = i + 1;
                    float x = input[i], y = input[j];
                    output[i] = x + y;
                }
            }
            """);

        var statements = module.EntryPoint.Body.Statements.Items;
        Assert.Collection(
            statements.Take(5),
            statement => AssertLocal(statement, "i"),
            statement => AssertLocal(statement, "j"),
            statement => AssertLocal(statement, "x"),
            statement => AssertLocal(statement, "y"),
            statement => Assert.IsType<ShaderAssignmentStatement>(statement));
    }

    [Fact]
    public void TypedIrLowererKeepsDynamicForLoopIndexExpressions()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct DynamicForKernel(ReadOnlyBuffer<int> counts, ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
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
            """);

        var loop = Assert.IsType<ShaderForStatement>(module.EntryPoint.Body.Statements.Items[2]);
        var condition = Assert.IsType<ShaderComparisonExpression>(loop.Condition);
        Assert.IsType<ShaderLocalReferenceExpression>(condition.Left);
        var dynamicBound = Assert.IsType<ShaderResourceElementExpression>(condition.Right);
        Assert.Equal("counts", dynamicBound.ResourceName);
        Assert.IsType<ShaderLocalReferenceExpression>(dynamicBound.Index);
        var bodyAssignment = Assert.Single(loop.Body.Statements.Items);
        var compound = Assert.IsType<ShaderCompoundAssignmentStatement>(bodyAssignment);
        var indexedInput = Assert.IsType<ShaderResourceElementExpression>(compound.Value);
        Assert.Equal("input", indexedInput.ResourceName);
        Assert.IsType<ShaderBinaryExpression>(indexedInput.Index);
    }

    [Fact]
    public void TypedIrLowererKeepsNumericCastsAsConversionExpressions()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct CastKernel(ReadOnlyBuffer<int> ints, ReadOnlyBuffer<float> floats, ReadWriteBuffer<float3> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float asFloat = ints[i];
                    int asInt = (int)floats[i];
                    float3 vector = new float3(ints[i], asInt, i);
                    output[i] = vector + new float3(asFloat);
                }
            }
            """);

        var statements = module.EntryPoint.Body.Statements.Items;
        var asFloat = Assert.IsType<ShaderLocalDeclarationStatement>(statements[1]);
        var intToFloat = Assert.IsType<ShaderConversionExpression>(asFloat.Initializer);
        Assert.Equal(ShaderTypeFactory.Float, intToFloat.Type);
        Assert.IsType<ShaderResourceElementExpression>(intToFloat.Operand);

        var asInt = Assert.IsType<ShaderLocalDeclarationStatement>(statements[2]);
        var floatToInt = Assert.IsType<ShaderConversionExpression>(asInt.Initializer);
        Assert.Equal(ShaderTypeFactory.Int, floatToInt.Type);
        Assert.IsType<ShaderResourceElementExpression>(floatToInt.Operand);

        var vector = Assert.IsType<ShaderLocalDeclarationStatement>(statements[3]);
        var constructor = Assert.IsType<ShaderConstructorExpression>(vector.Initializer);
        Assert.Equal(3, constructor.Arguments.Items.Count);
        Assert.IsType<ShaderConversionExpression>(constructor.Arguments.Items[0]);
        Assert.IsType<ShaderConversionExpression>(constructor.Arguments.Items[1]);
        Assert.IsType<ShaderConversionExpression>(constructor.Arguments.Items[2]);
    }

    [Fact]
    public void TypedIrLowererEmitsDoWhileBreakAndContinueStructurally()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct LoopFlowKernel(ReadWriteBuffer<int> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    do
                    {
                        i++;
                        if (i == 2)
                        {
                            continue;
                        }

                        if (i > 4)
                        {
                            break;
                        }
                    }
                    while (i < 8);

                    output[ThreadIds.X] = i;
                }
            }
            """);

        var doWhile = Assert.IsType<ShaderDoWhileStatement>(module.EntryPoint.Body.Statements.Items[1]);
        Assert.IsType<ShaderComparisonExpression>(doWhile.Condition);
        Assert.Contains(doWhile.Body.Statements.Items, statement => statement is ShaderIncrementDecrementStatement);
        var ifStatements = doWhile.Body.Statements.Items.OfType<ShaderIfStatement>().ToArray();
        Assert.Equal(2, ifStatements.Length);
        Assert.IsType<ShaderContinueStatement>(Assert.Single(ifStatements[0].Then.Statements.Items));
        Assert.IsType<ShaderBreakStatement>(Assert.Single(ifStatements[1].Then.Statements.Items));
    }

    [Fact]
    public void TypedIrLowererKeepsCallableAsModuleFunctionReference()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct CallableKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = Twice(input[i]);
                }

                [Callable]
                private static float Twice(float value)
                {
                    float doubled = value * 2;
                    return doubled;
                }
            }
            """);

        var callable = Assert.Single(module.Callables.Items);
        Assert.Equal("Twice", callable.Name);
        Assert.StartsWith("global__Scratch_CallableKernel_Twice_", callable.MangledName, StringComparison.Ordinal);
        Assert.Collection(callable.Body.Statements.Items,
            statement => AssertLocal(statement, "doubled"),
            statement => Assert.IsType<ShaderReturnStatement>(statement));
        var assignment = Assert.IsType<ShaderAssignmentStatement>(module.EntryPoint.Body.Statements.Items[1]);
        var call = Assert.IsType<ShaderCallableCallExpression>(assignment.Value);
        Assert.Equal(callable.MangledName, call.CallableName);
        Assert.Single(call.Arguments.Items);
    }

    [Fact]
    public void TypedIrLowererBindsOverloadedCallablesByMangledSymbolIdentity()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct CallableOverloadKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
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
            """);

        var callableNames = module.Callables.Items.Select(callable => callable.MangledName).ToArray();
        Assert.Equal(2, callableNames.Length);
        Assert.Equal(2, callableNames.Distinct(StringComparer.Ordinal).Count());

        var assignment = Assert.IsType<ShaderAssignmentStatement>(module.EntryPoint.Body.Statements.Items[1]);
        var call = Assert.IsType<ShaderCallableCallExpression>(assignment.Value);
        Assert.Contains(call.CallableName, callableNames);
        Assert.EndsWith("_float", call.CallableName, StringComparison.Ordinal);
        Assert.False(call.CallableName.EndsWith("_int", StringComparison.Ordinal), call.CallableName);
    }

    [Fact]
    public void TypedIrLowererRepresentsTextureSamplingStructurally()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct Rgba32
            {
                public float R;
                public float G;
                public float B;
                public float A;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct TextureSampleKernel(
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
            """);

        var assignment = Assert.IsType<ShaderAssignmentStatement>(module.EntryPoint.Body.Statements.Items[2]);
        var member = Assert.IsType<ShaderFieldReferenceExpression>(assignment.Value);
        var sample = Assert.IsType<ShaderTextureSampleExpression>(member.Instance);
        Assert.Equal(ShaderTextureSampleOperation.SampleLevel, sample.Operation);
        Assert.IsType<ShaderParameterReferenceExpression>(sample.Texture);
        Assert.IsType<ShaderParameterReferenceExpression>(sample.Sampler);
        Assert.IsType<ShaderLocalReferenceExpression>(sample.Uv);
        Assert.IsType<ShaderLiteralExpression>(sample.Lod);
    }

    [Fact]
    public void TypedIrLowererRepresentsSharedMemoryDeclarationAndElementAccessStructurally()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(4, 1, 1)]
            public readonly partial struct SharedKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
            {
                public void Execute()
                {
                    int localId = LocalIds.X;
                    int slot = localId + 2;
                    var shared = new SharedMemory<int>(8);
                    shared[slot] = input[localId];
                    GpuBarrier.Workgroup();
                    output[localId] = shared[slot];
                }
            }
            """);

        var statements = module.EntryPoint.Body.Statements.Items;
        var shared = Assert.IsType<ShaderSharedMemoryDeclarationStatement>(statements[2]);
        Assert.Equal("shared", shared.VariableName);
        Assert.Equal(8, shared.Length);
        Assert.Same(ShaderTypeFactory.Int, shared.ElementType);

        var write = Assert.IsType<ShaderAssignmentStatement>(statements[3]);
        var writeTarget = Assert.IsType<ShaderSharedMemoryElementLValue>(write.Target);
        Assert.Equal("shared", writeTarget.Name);
        Assert.IsType<ShaderLocalReferenceExpression>(writeTarget.Index);

        Assert.IsType<ShaderBarrierStatement>(statements[4]);

        var outputWrite = Assert.IsType<ShaderAssignmentStatement>(statements[5]);
        var read = Assert.IsType<ShaderSharedMemoryElementExpression>(outputWrite.Value);
        Assert.Equal("shared", read.Name);
        Assert.IsType<ShaderLocalReferenceExpression>(read.Index);
    }

    [Fact]
    public void TypedIrLowererPreservesMultipleSharedMemoryDeclarationsAndBarrierKinds()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(4, 1, 1)]
            public readonly partial struct SharedKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int localId = LocalIds.X;
                    var left = new SharedMemory<float>(4);
                    var right = new SharedMemory<int>(8);
                    left[localId] = input[localId];
                    right[localId] = localId;
                    GpuBarrier.Memory();
                    GpuBarrier.Full();
                    output[localId] = left[localId] + right[localId];
                }
            }
            """);

        var statements = module.EntryPoint.Body.Statements.Items;
        var left = Assert.IsType<ShaderSharedMemoryDeclarationStatement>(statements[1]);
        var right = Assert.IsType<ShaderSharedMemoryDeclarationStatement>(statements[2]);
        Assert.Equal("left", left.VariableName);
        Assert.Equal(4, left.Length);
        Assert.Same(ShaderTypeFactory.Float, left.ElementType);
        Assert.Equal("right", right.VariableName);
        Assert.Equal(8, right.Length);
        Assert.Same(ShaderTypeFactory.Int, right.ElementType);

        var memory = Assert.IsType<ShaderBarrierStatement>(statements[5]);
        var full = Assert.IsType<ShaderBarrierStatement>(statements[6]);
        Assert.Equal(ShaderBarrierKind.Memory, memory.Kind);
        Assert.Equal(ShaderBarrierKind.Full, full.Kind);
    }

    [Fact]
    public void TypedIrWriterSerializesSharedMemoryDeclarationAndDynamicElementAccess()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(4, 1, 1)]
            public readonly partial struct SharedKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
            {
                public void Execute()
                {
                    int localId = LocalIds.X;
                    int slot = localId + 2;
                    var shared = new SharedMemory<int>(8);
                    shared[slot] = input[localId];
                    GpuBarrier.Workgroup();
                    output[localId] = shared[slot];
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var sharedDeclaration = Assert.Single(section.Statements, statement => statement.Kind == 15);

        Assert.Equal("shared", section.Strings[(int)sharedDeclaration.NameId]);
        Assert.Equal(8u, sharedDeclaration.A);
        Assert.InRange(sharedDeclaration.Op, 0u, (uint)section.Types.Count - 1);

        var sharedWrite = Assert.Single(section.LValues, lvalue => lvalue.Kind == 9);
        Assert.Equal("shared", section.Strings[(int)sharedWrite.NameId]);
        Assert.InRange(sharedWrite.A, 0u, (uint)section.Expressions.Count - 1);
        Assert.Equal(2, section.Expressions[(int)sharedWrite.A].Kind);

        var sharedRead = Assert.Single(section.Expressions, expression => expression.Kind == 21);
        Assert.Equal("shared", section.Strings[(int)sharedRead.NameId]);
        Assert.InRange(sharedRead.A, 0u, (uint)section.Expressions.Count - 1);
        Assert.Equal(2, section.Expressions[(int)sharedRead.A].Kind);
    }

    [Fact]
    public void TypedIrWriterSerializesSharedMemoryVectorTypeAndBarrierKinds()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(2, 1, 1)]
            public readonly partial struct SharedVectorKernel(ReadWriteBuffer<float2> output) : IKernel1D
            {
                public void Execute()
                {
                    int localId = LocalIds.X;
                    var vectors = new SharedMemory<float2>(2);
                    vectors[localId] = new float2(localId, localId + 1);
                    GpuBarrier.Workgroup();
                    GpuBarrier.Memory();
                    GpuBarrier.Full();
                    output[localId] = vectors[localId];
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var sharedDeclaration = Assert.Single(section.Statements, statement => statement.Kind == 15);
        Assert.Equal("vectors", section.Strings[(int)sharedDeclaration.NameId]);
        Assert.Equal(2u, sharedDeclaration.A);

        var sharedType = section.Types[(int)sharedDeclaration.Op];
        Assert.Equal(2, sharedType.Kind);
        Assert.Equal(2u, sharedType.B);
        var vectorElementType = section.Types[(int)sharedType.A];
        Assert.Equal(1, vectorElementType.Kind);
        Assert.Equal((uint)ShaderPrimitiveKind.Float, vectorElementType.A);

        var barriers = section.Statements.Where(statement => statement.Kind == 13).Select(statement => statement.Op).ToArray();
        Assert.Equal([0u, 1u, 2u], barriers);
    }

    [Fact]
    public void TypedIrWriterSerializesAtomicExpressionsWithLValueTargets()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct AtomicKernel(ReadOnlyBuffer<int> input, ReadWriteBuffer<int> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    _ = GpuAtomic.Add(ref output[0], input[i]);
                    _ = GpuAtomic.CompareExchange(ref output[1], 0, input[i]);
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var atomics = section.Expressions.Where(expression => expression.Kind == 22).ToArray();

        Assert.Equal(2, atomics.Length);
        Assert.Contains(atomics, expression => expression.Op == (uint)ShaderAtomicOperation.Add &&
            expression.ArgumentCount == 1 &&
            expression.A < section.LValues.Count &&
            section.LValues[(int)expression.A].Kind == 4);
        Assert.Contains(atomics, expression => expression.Op == (uint)ShaderAtomicOperation.CompareExchange &&
            expression.ArgumentCount == 2 &&
            expression.A < section.LValues.Count &&
            section.LValues[(int)expression.A].Kind == 4);
        Assert.All(atomics, expression =>
        {
            Assert.InRange(expression.FirstArgument, 0u, (uint)section.Arguments.Count - expression.ArgumentCount);
            Assert.All(section.Arguments.Skip((int)expression.FirstArgument).Take((int)expression.ArgumentCount),
                argument => Assert.InRange(argument, 0u, (uint)section.Expressions.Count - 1));
        });
    }

    [Fact]
    public void TypedIrWriterSerializesLValueIndexExpressionsIntoExpressionTable()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct DynamicLValueKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    int j = i + 1;
                    output[j] = 42;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var assignment = section.Statements.Single(statement => statement.Kind == 3);
        var lvalue = section.LValues[(int)assignment.A];

        Assert.Equal(4, lvalue.Kind);
        Assert.NotEqual(uint.MaxValue, lvalue.A);
        Assert.InRange(lvalue.A, 0u, (uint)section.Expressions.Count - 1);
        Assert.Equal(2, section.Expressions[(int)lvalue.A].Kind);
    }

    [Fact]
    public void TypedIrWriterSerializesSwizzleLValuesStructurally()
    {
        var index = new ShaderLiteralExpression(ShaderTypeFactory.Int, "0");
        var vector = new ShaderResourceElementExpression(ShaderTypeFactory.Float4, "output", index, null!);
        var target = new ShaderSwizzleLValue(ShaderTypeFactory.Float2, vector, "XY");
        var value = new ShaderConstructorExpression(
            ShaderTypeFactory.Float2,
            new EquatableArray<ShaderExpression>([
                new ShaderLiteralExpression(ShaderTypeFactory.Float, "1.0"),
                new ShaderLiteralExpression(ShaderTypeFactory.Float, "2.0")
            ]));
        var body = new ShaderBlockStatement(new EquatableArray<ShaderStatement>([
            new ShaderAssignmentStatement(target, value)
        ]));
        var function = new ShaderFunctionModel(
            "Entry",
            "Entry",
            ShaderFunctionKind.Compute1D,
            ShaderTypeFactory.Void,
            new EquatableArray<ShaderParameterModel>(),
            body);
        var module = new ShaderModuleModel(
            function,
            new EquatableArray<ShaderFunctionModel>(),
            new EquatableArray<ResourceModel>([
                new ResourceModel(0, "output", "ReadWriteBuffer<float4>", "Feather.Math.float4", ResourceKindModel.Buffer, ResourceAccessModel.ReadWrite)
            ]),
            new EquatableArray<ShaderStructType>(),
            new ThreadGroupModel(1, 1, 1),
            "Entry",
            "Scratch");

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var assignment = section.Statements.Single(statement => statement.Kind == 3);
        var lvalue = section.LValues[(int)assignment.A];

        Assert.Equal(5, lvalue.Kind);
        Assert.Equal("XY", section.Strings[(int)lvalue.NameId]);
        Assert.InRange(lvalue.A, 0u, (uint)section.Expressions.Count - 1);
        var vectorExpression = section.Expressions[(int)lvalue.A];
        Assert.Equal(5, vectorExpression.Kind);
        Assert.Equal("output", section.Strings[(int)vectorExpression.NameId]);
    }

    [Fact]
    public void TypedIrWriterSerializesBoolVectorTypesStructurally()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BoolVectorKernel(ReadWriteBuffer<int> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    bool2 pair = new bool2(i == 0, i != 1);
                    bool3 triple = new bool3(pair.X, pair.Y, i < 2);
                    bool4 quad = new bool4(triple.X, triple.Y, triple.Z, i > 2);
                    output[i] = quad.X ? 1 : 0;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var boolPrimitiveIndex = section.Types
            .Select((type, index) => (type, index))
            .First(candidate => candidate.type.Kind == 1 && candidate.type.A == (uint)ShaderPrimitiveKind.Bool)
            .index;

        Assert.InRange(boolPrimitiveIndex, 0, section.Types.Count - 1);
        Assert.Contains(section.Types, type => type.Kind == 2 && type.A == (uint)boolPrimitiveIndex && type.B == 2);
        Assert.Contains(section.Types, type => type.Kind == 2 && type.A == (uint)boolPrimitiveIndex && type.B == 3);
        Assert.Contains(section.Types, type => type.Kind == 2 && type.A == (uint)boolPrimitiveIndex && type.B == 4);
        Assert.Contains(section.Expressions, expression => expression.Kind == 12 &&
            section.Types[(int)expression.TyId].Kind == 2 &&
            section.Types[(int)expression.TyId].A == (uint)boolPrimitiveIndex);
        Assert.Contains(section.Expressions, expression => expression.Kind == 15 &&
            section.Types[(int)expression.TyId].Kind == 1 &&
            section.Types[(int)expression.TyId].A == (uint)ShaderPrimitiveKind.Bool);
    }

    [Fact]
    public void TypedIrWriterSerializesTextureCoordinateSwizzleAliasesStructurally()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct TextureCoordinateSwizzleKernel(ReadOnlyBuffer<float4> input, ReadWriteBuffer<float4> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float4 value = input[i];
                    output[i] = new float4(value.TS, value.P, value.QPTS.W);
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var swizzles = section.Expressions
            .Where(static expression => expression.Kind == 15)
            .Select(expression => section.Strings[(int)expression.NameId])
            .ToArray();

        Assert.Contains("TS", swizzles);
        Assert.Contains("P", swizzles);
        Assert.Contains("QPTS", swizzles);
    }

    [Fact]
    public void TypedIrWriterSerializesVectorMathIntrinsicCalls()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct VectorIntrinsicKernel(ReadOnlyBuffer<float3> left, ReadOnlyBuffer<float3> right, ReadWriteBuffer<float3> output) : IKernel1D
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
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var intrinsicNames = section.Expressions
            .Where(expression => expression.Kind == 13)
            .Select(expression => section.Strings[(int)expression.NameId])
            .ToArray();

        Assert.Contains("global::Feather.Math.ShaderMath.Abs", intrinsicNames);
        Assert.Contains("global::Feather.Math.ShaderMath.Min", intrinsicNames);
        Assert.Contains("global::Feather.Math.ShaderMath.Clamp", intrinsicNames);
        Assert.Contains("global::Feather.Math.ShaderMath.Mix", intrinsicNames);
        Assert.All(section.Expressions.Where(expression => expression.Kind == 13), expression =>
        {
            Assert.NotEqual(uint.MaxValue, expression.FirstArgument);
            Assert.InRange(expression.FirstArgument, 0u, (uint)section.Arguments.Count - expression.ArgumentCount);
        });
    }

    [Fact]
    public void TypedIrWriterSerializesTextureSamplingAsDedicatedExpressionKind()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct Rgba32
            {
                public float R;
                public float G;
                public float B;
                public float A;
            }

            [Kernel]
            [ThreadGroupSize(1)]
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
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var sample = Assert.Single(section.Expressions, expression => expression.Kind == 23);
        Assert.Equal(0u, sample.Op);
        Assert.Equal(3u, sample.ArgumentCount);
        Assert.NotEqual(uint.MaxValue, sample.FirstArgument);
        Assert.DoesNotContain(section.Expressions.Where(expression => expression.Kind == 13), expression =>
            expression.NameId < section.Strings.Count &&
            section.Strings[(int)expression.NameId].Contains("Sample", StringComparison.Ordinal));
        Assert.All(section.Arguments.Skip((int)sample.FirstArgument).Take((int)sample.ArgumentCount), argument =>
            Assert.InRange(argument, 0u, (uint)section.Expressions.Count - 1));
    }

    [Fact]
    public void TypedIrWriterSerializesMatrixMathIntrinsicCalls()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct MatrixIntrinsicKernel(ReadWriteBuffer<float4> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float4x4 a = new float4x4(
                        new float4(1, 0, 0, 0),
                        new float4(0, 2, 0, 0),
                        new float4(0, 0, 4, 0),
                        new float4(0, 0, 0, 1));
                    float4x4 b = ShaderMath.Inverse(a);
                    float4x4 c = ShaderMath.Hadamard(ShaderMath.Transpose(a), b);
                    output[i] = Hlsl.Mul(c, new float4(ShaderMath.Determinant(a), 1, 1, 1));
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var intrinsicNames = section.Expressions
            .Where(expression => expression.Kind == 13)
            .Select(expression => section.Strings[(int)expression.NameId])
            .ToArray();

        Assert.Contains("global::Feather.Math.ShaderMath.Inverse", intrinsicNames);
        Assert.Contains("global::Feather.Math.ShaderMath.Hadamard", intrinsicNames);
        Assert.Contains("global::Feather.Math.ShaderMath.Transpose", intrinsicNames);
        Assert.Contains("global::Feather.Math.ShaderMath.Determinant", intrinsicNames);
        Assert.Contains("global::Feather.Math.Hlsl.Mul", intrinsicNames);
    }

    [Fact]
    public void TypedIrWriterUsesVersionedHeaderAndExplicitFunctionBlockAndArgumentRanges()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct CallableKernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
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
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));

        Assert.Equal("FTIR", section.Magic);
        Assert.Equal(1, section.MajorVersion);
        Assert.Equal(1, section.MinorVersion);
        Assert.Equal(104, section.HeaderSize);
        Assert.Equal(0u, section.EntryFunctionId);
        Assert.Equal(2, section.Functions.Count);
        Assert.DoesNotContain(section.Statements, statement => statement.Kind == 100);

        var entry = section.Functions[0];
        Assert.Equal((byte)ShaderFunctionKind.Compute1D, entry.Kind);
        Assert.Equal(0u, entry.ParameterCount);
        Assert.Equal(uint.MaxValue, entry.FirstParameter);
        var entryBlock = section.Statements[(int)entry.BodyStatementId];
        Assert.Equal(1, entryBlock.Kind);
        Assert.Equal(2u, entryBlock.ChildCount);
        Assert.InRange(entryBlock.FirstChild, 0u, (uint)section.Children.Count - entryBlock.ChildCount);

        var callable = section.Functions[1];
        Assert.Equal((byte)ShaderFunctionKind.Callable, callable.Kind);
        Assert.Equal(1u, callable.ParameterCount);
        Assert.Equal(0u, callable.FirstParameter);
        Assert.Equal("value", section.Strings[(int)section.Parameters[(int)callable.FirstParameter].NameId]);

        var callableBlock = section.Statements[(int)callable.BodyStatementId];
        Assert.Equal(1, callableBlock.Kind);
        Assert.Equal(1u, callableBlock.ChildCount);
        Assert.InRange(callableBlock.FirstChild, 0u, (uint)section.Children.Count - callableBlock.ChildCount);

        var call = section.Expressions.Single(expression => expression.Kind == 14);
        Assert.Equal(1u, call.ArgumentCount);
        Assert.InRange(call.FirstArgument, 0u, (uint)section.Arguments.Count - call.ArgumentCount);
        Assert.InRange(section.Arguments[(int)call.FirstArgument], 0u, (uint)section.Expressions.Count - 1);
    }

    [Fact]
    public void TypedIrWriterSerializesEmptyBlockChildRangesWithoutGuessing()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct EmptyBlockKernel(ReadWriteBuffer<int> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    if (i > 0)
                    {
                    }
                    else
                    {
                    }

                    output[i] = i;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var ifStatement = section.Statements.Single(statement => statement.Kind == 5);
        var thenBlock = section.Statements[(int)ifStatement.B];
        var elseBlock = section.Statements[(int)ifStatement.C];

        Assert.Equal(1, thenBlock.Kind);
        Assert.Equal(uint.MaxValue, thenBlock.FirstChild);
        Assert.Equal(0u, thenBlock.ChildCount);
        Assert.Equal(1, elseBlock.Kind);
        Assert.Equal(uint.MaxValue, elseBlock.FirstChild);
        Assert.Equal(0u, elseBlock.ChildCount);
    }

    [Fact]
    public void TypedIrWriterSerializesZeroArgumentCallableWithEmptyArgumentSpan()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct ZeroArgCallableKernel(ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    output[ThreadIds.X] = FortyTwo();
                }

                [Callable]
                private static float FortyTwo() => 42;
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var call = section.Expressions.Single(expression => expression.Kind == 14);

        Assert.Equal(uint.MaxValue, call.FirstArgument);
        Assert.Equal(0u, call.ArgumentCount);
    }

    [Fact]
    public void TypedIrWriterEncodesAllTypeVariantsAndStructFieldLayout()
    {
        var byteType = new ShaderPrimitiveType(ShaderPrimitiveKind.UInt) { CSharpTypeName = "byte" };
        var packed = new ShaderStructType(
            "Packed",
            "global::Scratch.Packed",
            new EquatableArray<ShaderStructField>([
                new ShaderStructField("Flags", byteType, 0, 1),
                new ShaderStructField("Position", ShaderTypeFactory.Float3, 16, 12)
            ]),
            32,
            16)
        {
            CSharpTypeName = "global::Scratch.Packed"
        };
        var arrayType = new ShaderArrayType(ShaderTypeFactory.Int, 4) { CSharpTypeName = "int[]" };
        var resourceType = new ShaderResourceWrapperType(
            ShaderResourceKind.Buffer,
            ShaderTypeFactory.Float,
            ShaderResourceAccess.Read)
        {
            CSharpTypeName = "ReadOnlyBuffer<float>"
        };
        var body = new ShaderBlockStatement(new EquatableArray<ShaderStatement>([
            new ShaderLocalDeclarationStatement("m", ShaderTypeFactory.Float2x2, null, null!)
        ]));
        var function = new ShaderFunctionModel(
            "Entry",
            "Entry",
            ShaderFunctionKind.Compute1D,
            ShaderTypeFactory.Void,
            new EquatableArray<ShaderParameterModel>([
                new ShaderParameterModel("items", resourceType, ShaderParameterDirection.In),
                new ShaderParameterModel("scratch", arrayType, ShaderParameterDirection.InOut),
                new ShaderParameterModel("mask", ShaderTypeFactory.UInt, ShaderParameterDirection.In),
                new ShaderParameterModel("packed", packed, ShaderParameterDirection.In)
            ]),
            body);
        var module = new ShaderModuleModel(
            function,
            new EquatableArray<ShaderFunctionModel>(),
            new EquatableArray<ResourceModel>(),
            new EquatableArray<ShaderStructType>([packed]),
            new ThreadGroupModel(1, 1, 1),
            "Entry",
            "Scratch");

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));

        Assert.Contains(section.Types, type => type.Kind == 1 && type.A == (uint)ShaderPrimitiveKind.UInt && type.B == 8);
        Assert.Contains(section.Types, type => type.Kind == 1 && type.A == (uint)ShaderPrimitiveKind.UInt && type.B == 32);
        Assert.Contains(section.Types, type => type.Kind == 2);
        Assert.Contains(section.Types, type => type.Kind == 3);
        Assert.Contains(section.Types, type => type.Kind == 4);
        Assert.Contains(section.Types, type => type.Kind == 5 && type.B == 4);
        Assert.Contains(section.Types, type => type.Kind == 6 && type.A == (uint)ShaderResourceKind.Buffer && type.C == (uint)ShaderResourceAccess.Read);
        Assert.Contains(section.Types, type => type.Kind == 7);

        var structure = Assert.Single(section.Structs);
        Assert.Equal(32u, structure.SizeInBytes);
        Assert.Equal(16u, structure.Alignment);
        Assert.Equal(2u, structure.FieldCount);
        Assert.InRange(structure.FirstField, 0u, (uint)section.StructFields.Count - structure.FieldCount);

        var firstField = section.StructFields[(int)structure.FirstField];
        var secondField = section.StructFields[(int)structure.FirstField + 1];
        Assert.Equal("Flags", section.Strings[(int)firstField.NameId]);
        Assert.Equal(0u, firstField.Offset);
        Assert.Equal(1u, firstField.SizeInBytes);
        Assert.Equal("Position", section.Strings[(int)secondField.NameId]);
        Assert.Equal(16u, secondField.Offset);
        Assert.Equal(12u, secondField.SizeInBytes);
    }

    [Fact]
    public void TypedIrWriterEncodesFloat2x2GpuStructFieldWithEasyGpuStd430Layout()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct MatrixScene
            {
                public float Weight;
                public float2x2 Transform;
                public float Bias;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct MatrixSceneKernel(ReadOnlyBuffer<MatrixScene> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    MatrixScene scene = input[i];
                    output[i] = scene.Weight + scene.Bias;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var structure = Assert.Single(section.Structs);

        Assert.Equal(64u, structure.SizeInBytes);
        Assert.Equal(16u, structure.Alignment);
        Assert.Equal(3u, structure.FieldCount);
        Assert.InRange(structure.FirstField, 0u, (uint)section.StructFields.Count - structure.FieldCount);

        var fields = section.StructFields
            .Skip((int)structure.FirstField)
            .Take((int)structure.FieldCount)
            .ToArray();

        AssertField("Weight", 0, 4, fields[0], section);
        AssertField("Transform", 16, 32, fields[1], section);
        AssertField("Bias", 48, 4, fields[2], section);

        static void AssertField(string name, uint offset, uint size, TypedIrStructField field, TypedIrSection section)
        {
            Assert.Equal(name, section.Strings[(int)field.NameId]);
            Assert.Equal(offset, field.Offset);
            Assert.Equal(size, field.SizeInBytes);
        }
    }

    [Fact]
    public void TypedIrWriterEncodesBoolVectorGpuStructFieldWithEasyGpuStd430Layout()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct BoolVectorScene
            {
                public bool Enabled;
                public bool3 Mask;
                public float Weight;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BoolVectorSceneKernel(ReadOnlyBuffer<BoolVectorScene> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    BoolVectorScene scene = input[i];
                    output[i] = scene.Weight;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var structure = Assert.Single(section.Structs);

        Assert.Equal(32u, structure.SizeInBytes);
        Assert.Equal(16u, structure.Alignment);
        Assert.Equal(3u, structure.FieldCount);
        Assert.InRange(structure.FirstField, 0u, (uint)section.StructFields.Count - structure.FieldCount);

        var fields = section.StructFields
            .Skip((int)structure.FirstField)
            .Take((int)structure.FieldCount)
            .ToArray();

        AssertField("Enabled", 0, 4, fields[0], section);
        AssertField("Mask", 16, 12, fields[1], section);
        AssertField("Weight", 28, 4, fields[2], section);

        static void AssertField(string name, uint offset, uint size, TypedIrStructField field, TypedIrSection section)
        {
            Assert.Equal(name, section.Strings[(int)field.NameId]);
            Assert.Equal(offset, field.Offset);
            Assert.Equal(size, field.SizeInBytes);
        }
    }

    [Fact]
    public void TypedIrWriterEncodesNestedGpuStructFieldLayout()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct InnerScene
            {
                public float3 Direction;
                public float Intensity;
            }

            [GpuStruct]
            public partial struct OuterScene
            {
                public InnerScene Inner;
                public float4 Weight;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct NestedSceneKernel(ReadOnlyBuffer<OuterScene> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    OuterScene scene = input[i];
                    output[i] = scene.Inner.Intensity + scene.Inner.Direction.X + scene.Weight.X;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var structs = section.Structs.ToDictionary(
            structure => section.Strings[(int)structure.NameId],
            StringComparer.Ordinal);

        var inner = structs["InnerScene"];
        var outer = structs["OuterScene"];
        Assert.Equal(16u, inner.SizeInBytes);
        Assert.Equal(16u, inner.Alignment);
        Assert.Equal(32u, outer.SizeInBytes);
        Assert.Equal(16u, outer.Alignment);

        var innerFields = section.StructFields.Skip((int)inner.FirstField).Take((int)inner.FieldCount).ToArray();
        var outerFields = section.StructFields.Skip((int)outer.FirstField).Take((int)outer.FieldCount).ToArray();

        AssertField("Direction", 0, 12, innerFields[0], section);
        AssertField("Intensity", 12, 4, innerFields[1], section);
        AssertField("Inner", 0, 16, outerFields[0], section);
        AssertField("Weight", 16, 16, outerFields[1], section);

        static void AssertField(string name, uint offset, uint size, TypedIrStructField field, TypedIrSection section)
        {
            Assert.Equal(name, section.Strings[(int)field.NameId]);
            Assert.Equal(offset, field.Offset);
            Assert.Equal(size, field.SizeInBytes);
        }
    }

    [Fact]
    public void TypedIrLowererUsesCanonicalRecordGpuStructFieldIndex()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public readonly partial record struct PackedColor(uint R, uint G, uint B, uint A);

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct RecordFieldKernel(ReadOnlyBuffer<PackedColor> input, ReadWriteBuffer<uint> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    output[i] = input[i].R;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var structure = Assert.Single(section.Structs);
        var field = section.StructFields[(int)structure.FirstField];

        Assert.Equal("R", section.Strings[(int)field.NameId]);
        Assert.Equal(0u, field.Offset);
        Assert.Contains(section.Expressions, expression => expression.Kind == 16 && section.Strings[(int)expression.NameId] == "R");
    }

    [Fact]
    public void TypedIrLowererRejectsInvalidGpuStructBeforeStructRecordEmission()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct BadLayout
            {
                public readonly float Value;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadOnlyBuffer<BadLayout> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    BadLayout value = input[i];
                    output[i] = 0.0f;
                }
            }
            """);

        var syntaxTree = compilation.SyntaxTrees.Single();
        var syntax = syntaxTree.GetRoot().DescendantNodes().OfType<StructDeclarationSyntax>().Single(static s =>
            s.AttributeLists.SelectMany(static list => list.Attributes).Any(static attribute =>
                attribute.Name.ToString() is "Kernel" or "KernelAttribute"));
        var semanticModel = compilation.GetSemanticModel(syntaxTree);
        var symbol = semanticModel.GetDeclaredSymbol(syntax)!;
        var model = ShaderModelFactory.Create(syntax, symbol)!;

        var exception = Assert.Throws<ShaderIrLoweringException>(() =>
            ShaderIrLowerer.Lower(model, semanticModel, CancellationToken.None));

        Assert.Contains("BadLayout", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void GeneratorEmitsAdParameterAndLossMetadata()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct LossKernel(ReadWriteBuffer<float> parameters, ReadWriteBuffer<float> loss) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float p = parameters[i];
                    float l = p * p;
                    loss[i] = l;
                    AD.Parameter(parameters[i]);
                    AD.Loss(l);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("LossKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var parameter = Assert.Single(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Parameter);
        Assert.Equal("parameters", parameter.Name);
        Assert.Equal("parameters", parameter.ResourceName);
        Assert.Equal(0u, parameter.Binding);
        Assert.Equal("float", parameter.TypeName);
        Assert.Equal("i", parameter.Index);
        Assert.Equal(FeatherIrAdSourceKind.BufferElement, parameter.SourceKind);

        var loss = Assert.Single(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Loss);
        Assert.Equal("l", loss.Name);
        Assert.Equal(string.Empty, loss.ResourceName);
        Assert.Equal(uint.MaxValue, loss.Binding);
        Assert.Equal("float", loss.TypeName);
        Assert.Equal(FeatherIrAdSourceKind.Local, loss.SourceKind);

        Assert.DoesNotContain(module.Instructions, instruction =>
            instruction.Opcode == FeatherIrInstructionOpcode.Invocation &&
            instruction.Operand.Contains("Feather.AD.AD", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorEmitsAdParameterMetadataForConstantBufferIndex()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct ScalarParameterKernel(ReadOnlyBuffer<float> x, ReadWriteBuffer<float> w, ReadWriteBuffer<float> loss) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float pred = w[0] * x[i];
                    loss[i] = pred * pred;
                    AD.Parameter(w[0]);
                    AD.Loss(pred);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("ScalarParameterKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var parameter = Assert.Single(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Parameter);
        Assert.Equal("w", parameter.Name);
        Assert.Equal("w", parameter.ResourceName);
        Assert.Equal(1u, parameter.Binding);
        Assert.Equal("float", parameter.TypeName);
        Assert.Equal("0", parameter.Index);
        Assert.Equal(FeatherIrAdSourceKind.BufferElement, parameter.SourceKind);
    }

    [Fact]
    public void GeneratorEmitsDeterministicAdMetadataForNnLinearTrainingKernel()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct NnLinearTrainingKernel(
                ReadWriteBuffer<float> weight,
                ReadWriteBuffer<float> bias,
                ReadOnlyBuffer<float> x,
                ReadOnlyBuffer<float> y,
                ReadWriteBuffer<float> loss) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float prediction = (weight[0] * x[i]) + bias[0];
                    float error = prediction - y[i];
                    float l = error * error;
                    loss[i] = l;
                    AD.Parameter(weight[0]);
                    AD.Parameter(bias[0]);
                    AD.Loss(l);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("NnLinearTrainingKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());
        var parameters = module.AdAnnotations
            .Where(annotation => annotation.Role == FeatherIrAdAnnotationRole.Parameter)
            .OrderBy(annotation => annotation.Binding)
            .ToArray();

        Assert.Equal(2, parameters.Length);
        Assert.Equal("weight", parameters[0].Name);
        Assert.Equal("weight", parameters[0].ResourceName);
        Assert.Equal(0u, parameters[0].Binding);
        Assert.Equal("0", parameters[0].Index);
        Assert.Equal("bias", parameters[1].Name);
        Assert.Equal("bias", parameters[1].ResourceName);
        Assert.Equal(1u, parameters[1].Binding);
        Assert.Equal("0", parameters[1].Index);
        Assert.Single(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Loss);
    }

    [Fact]
    public void GeneratorEmitsAdLossMetadataForDirectBufferElement()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct DirectLossKernel(ReadWriteBuffer<float> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    AD.Parameter(parameters[i]);
                    AD.Loss(parameters[i]);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("DirectLossKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var loss = Assert.Single(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Loss);
        Assert.Equal("parameters", loss.Name);
        Assert.Equal("parameters", loss.ResourceName);
        Assert.Equal(0u, loss.Binding);
        Assert.Equal("i", loss.Index);
        Assert.Equal(FeatherIrAdSourceKind.BufferElement, loss.SourceKind);
    }

    [Fact]
    public async Task AnalyzerRejectsAdMarkerOutsideGeneratedKernel()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather.AD;

            public static class CpuCode
            {
                public static void Run(float value)
                {
                    AD.Parameter(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0021");
    }

    [Fact]
    public async Task AnalyzerRejectsAdMarkerWithoutAutoDiffAttribute()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float value = parameters[i];
                    AD.Parameter(parameters[i]);
                    AD.Loss(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0021"
            && diagnostic.GetMessage().Contains("[AutoDiff]", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsAdMarkerInGraphicsShader()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Math;
            using Feather.Resources;

            [VertexShader]
            [AutoDiff]
            public readonly partial struct BadVertex(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
            {
                public float4 Execute()
                {
                    float4 position = vertices[0];
                    AD.Parameter(position);
                    AD.Loss(position.X);
                    return position;
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0021"
            && diagnostic.GetMessage().Contains("one-dimensional compute", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsMultipleAdLossMarkers()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float value = parameters[i];
                    AD.Parameter(parameters[i]);
                    AD.Loss(value);
                    AD.Loss(value * value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0021"
            && diagnostic.GetMessage().Contains("only one scalar loss", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsNonScalarAdLoss()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<float2> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float2 value = parameters[i];
                    AD.Parameter(parameters[i]);
                    AD.Loss(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("loss must be a scalar float", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorEmitsAdParameterMetadataForTraceableLocalAlias()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct AliasKernel(ReadWriteBuffer<float> parameters, ReadWriteBuffer<float> loss) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float value = parameters[i];
                    float l = value * value;
                    loss[i] = l;
                    AD.Parameter(value);
                    AD.Loss(l);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("AliasKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var parameter = Assert.Single(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Parameter);
        Assert.Equal("parameters", parameter.Name);
        Assert.Equal("parameters", parameter.ResourceName);
        Assert.Equal(0u, parameter.Binding);
        Assert.Equal("float", parameter.TypeName);
        Assert.Equal("i", parameter.Index);
        Assert.Equal(FeatherIrAdSourceKind.BufferElement, parameter.SourceKind);
    }

    [Fact]
    public void GeneratorEmitsAdParameterMetadataForConstantIndexLocalAlias()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct ConstantAliasKernel(ReadWriteBuffer<float> parameters, ReadWriteBuffer<float> loss) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float value = parameters[0];
                    float l = value * value;
                    loss[i] = l;
                    AD.Parameter(value);
                    AD.Loss(l);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("ConstantAliasKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var parameter = Assert.Single(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Parameter);
        Assert.Equal("parameters", parameter.ResourceName);
        Assert.Equal(0u, parameter.Binding);
        Assert.Equal("0", parameter.Index);
        Assert.Equal(FeatherIrAdSourceKind.BufferElement, parameter.SourceKind);
    }

    [Fact]
    public void GeneratorTracesParenthesizedCastAdParameterAlias()
    {
        var compilation = CreateCompilation("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct CastAliasKernel(ReadWriteBuffer<float> parameters, ReadWriteBuffer<float> loss) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float value = (float)(parameters[i]);
                    float l = value * value;
                    loss[i] = l;
                    AD.Parameter((float)(value));
                    AD.Loss(l);
                }
            }
            """);

        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var outputCompilation, out var diagnostics);

        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        var generated = outputCompilation.SyntaxTrees.Single(tree => tree.FilePath.EndsWith("CastAliasKernel.Feather.g.cs", StringComparison.Ordinal));
        var module = ReadGeneratedIr(generated.ToString());

        var parameter = Assert.Single(module.AdAnnotations, annotation => annotation.Role == FeatherIrAdAnnotationRole.Parameter);
        Assert.Equal("parameters", parameter.ResourceName);
        Assert.Equal("i", parameter.Index);
        Assert.Equal(FeatherIrAdSourceKind.BufferElement, parameter.SourceKind);
    }

    [Fact]
    public async Task AnalyzerRejectsMutatedAdParameterAlias()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> parameters) : IKernel1D
            {
                public void Execute()
                {
                    float value = parameters[ThreadIds.X];
                    value = value + 1f;
                    AD.Parameter(value);
                    AD.Loss(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("directly traceable", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsIncrementedAdParameterAlias()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> parameters) : IKernel1D
            {
                public void Execute()
                {
                    float value = parameters[ThreadIds.X];
                    value++;
                    AD.Parameter(value);
                    AD.Loss(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("directly traceable", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsDecrementedAdParameterAlias()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> parameters) : IKernel1D
            {
                public void Execute()
                {
                    float value = parameters[ThreadIds.X];
                    value--;
                    AD.Parameter(value);
                    AD.Loss(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("directly traceable", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsUnsupportedAdTextureParameter()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadOnlyTexture2D<float> input) : IKernel1D
            {
                public void Execute()
                {
                    float value = input[default];
                    AD.Parameter(input[default]);
                    AD.Loss(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022");
    }

    [Fact]
    public async Task AnalyzerRejectsUntraceableAdParameterExpression()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float value = parameters[i] + 1f;
                    AD.Parameter(value);
                    AD.Loss(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("directly traceable", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsUnsupportedAdParameterType()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<int> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    int value = parameters[i];
                    AD.Parameter(value);
                    AD.Loss(1f);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("parameter type", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("uint", "value > 0u")]
    [InlineData("bool", "value")]
    public async Task AnalyzerRejectsUnsupportedScalarAdParameterTypes(string parameterType, string lossExpression)
    {
        var diagnostics = await AnalyzeAsync($$"""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<{{parameterType}}> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    {{parameterType}} value = parameters[i];
                    AD.Parameter(value);
                    AD.Loss({{lossExpression}} ? 1f : 0f);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("parameter type", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsStructAdParameter()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [GpuStruct]
            public partial struct Pair
            {
                public float X;
                public float Y;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<Pair> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    Pair value = parameters[i];
                    AD.Parameter(value);
                    AD.Loss(value.X);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("parameter type", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsMatrixAdParameter()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Math;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<float2x2> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float2x2 value = parameters[i];
                    AD.Parameter(value);
                    AD.Loss(value.M11);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("parameter type", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsSamplerDerivedAdParameter()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Math;
            using Feather.Resources;

            [GpuStruct]
            public partial struct Rgba32
            {
                public float R;
                public float G;
                public float B;
                public float A;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(SampledTexture2D<Rgba32> input, SamplerState sampler) : IKernel1D
            {
                public void Execute()
                {
                    float value = input.Sample(sampler, new float2(0.5f, 0.5f)).R;
                    AD.Parameter(value);
                    AD.Loss(value);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("directly traceable", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsAtomicDerivedAdParameter()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<int> counters) : IKernel1D
            {
                public void Execute()
                {
                    int value = GpuAtomic.Add(counters[0], 1);
                    AD.Parameter(value);
                    AD.Loss(1f);
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0022"
            && diagnostic.GetMessage().Contains("parameter type", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AnalyzerRejectsNestedCallableAd()
    {
        var diagnostics = await AnalyzeAsync("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct BadKernel(ReadWriteBuffer<float> parameters) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float p = parameters[i];
                    float y = Outer(p);
                    AD.Parameter(parameters[i]);
                    AD.Loss(y * y);
                }

                [Callable]
                private static float Outer(float value)
                {
                    return Inner(value) * 3f;
                }

                [Callable]
                private static float Inner(float value)
                {
                    return value * value;
                }
            }
            """);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Id == "FE0021"
            && diagnostic.GetMessage().Contains("nested callable-to-callable AD", StringComparison.Ordinal));
    }

    [Fact]
    public void TypedIrWriterSerializesAdControlFlowAndMarkerSkippingPaths()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.AD;
            using Feather.Resources;

            namespace Scratch;

            [Kernel]
            [ThreadGroupSize(1)]
            [AutoDiff]
            public readonly partial struct AdControlFlowKernel(ReadWriteBuffer<float> parameters, ReadWriteBuffer<float> loss) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    float p = parameters[i];
                    float sum = 0f;
                    for (int j = 1; j < 5; j = j + 2)
                    {
                        sum += p;
                    }

                    while (sum < 10f)
                    {
                        sum = sum + 1f;
                    }

                    do
                    {
                        sum = sum - 1f;
                        if (sum < 2f)
                        {
                            continue;
                        }

                        if (sum > 8f)
                        {
                            break;
                        }
                    }
                    while (sum > 0f);

                    loss[i] = sum * sum;
                    AD.Parameter(parameters[i]);
                    AD.Loss(loss[i]);
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));

        Assert.Contains(section.Statements, statement => statement.Kind == 4);
        Assert.Contains(section.Statements, statement => statement.Kind == 6);
        Assert.Contains(section.Statements, statement => statement.Kind == 7);
        Assert.Contains(section.Statements, statement => statement.Kind == 8);
        Assert.Contains(section.Statements, statement => statement.Kind == 9);
        Assert.Contains(section.Statements, statement => statement.Kind == 10);
        Assert.DoesNotContain(section.Expressions, expression =>
            expression.Kind == 14 &&
            expression.NameId < section.Strings.Count &&
            section.Strings[(int)expression.NameId].Contains("Feather.AD.AD", StringComparison.Ordinal));
    }

    [Fact]
    public void TypedIrWriterEncodesGpuStructArrayFieldLayout()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Math;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct ArrayScene
            {
                public GpuArray4<float3> Directions;
                public float Weight;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct ArraySceneKernel(ReadOnlyBuffer<ArrayScene> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    ArrayScene scene = input[i];
                    output[i] = scene.Directions[2].X + scene.Weight;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var structure = Assert.Single(section.Structs);

        Assert.Equal(80u, structure.SizeInBytes);
        Assert.Equal(16u, structure.Alignment);
        Assert.Equal(2u, structure.FieldCount);

        var fields = section.StructFields
            .Skip((int)structure.FirstField)
            .Take((int)structure.FieldCount)
            .ToArray();
        var directions = fields[0];
        var weight = fields[1];
        var directionsType = section.Types[(int)directions.TypeId];
        var elementType = section.Types[(int)directionsType.A];

        Assert.Equal("Directions", section.Strings[(int)directions.NameId]);
        Assert.Equal(0u, directions.Offset);
        Assert.Equal(64u, directions.SizeInBytes);
        Assert.Equal(5, directionsType.Kind);
        Assert.Equal(4u, directionsType.B);
        Assert.Equal(2, elementType.Kind);
        Assert.Equal(3u, elementType.B);
        Assert.Equal("Weight", section.Strings[(int)weight.NameId]);
        Assert.Equal(64u, weight.Offset);
        Assert.Equal(4u, weight.SizeInBytes);
    }

    [Fact]
    public void TypedIrWriterEncodesNestedGpuStructArrayFieldLayout()
    {
        var module = LowerTypedModule("""
            using Feather;
            using Feather.Resources;

            namespace Scratch;

            [GpuStruct]
            public partial struct InnerScene
            {
                public float Value;
            }

            [GpuStruct]
            public partial struct OuterScene
            {
                public GpuArray3<InnerScene> Items;
                public float Weight;
            }

            [Kernel]
            [ThreadGroupSize(1)]
            public readonly partial struct NestedArraySceneKernel(ReadOnlyBuffer<OuterScene> input, ReadWriteBuffer<float> output) : IKernel1D
            {
                public void Execute()
                {
                    int i = ThreadIds.X;
                    OuterScene scene = input[i];
                    output[i] = scene.Items[1].Value + scene.Weight;
                }
            }
            """);

        var section = ReadTypedIrSection(ShaderIrModuleWriter.WriteModule(module));
        var structs = section.Structs.ToDictionary(
            structure => section.Strings[(int)structure.NameId],
            StringComparer.Ordinal);
        var inner = structs["InnerScene"];
        var outer = structs["OuterScene"];
        var outerFields = section.StructFields.Skip((int)outer.FirstField).Take((int)outer.FieldCount).ToArray();
        var items = outerFields[0];
        var itemsType = section.Types[(int)items.TypeId];
        var itemElementType = section.Types[(int)itemsType.A];

        Assert.Equal(4u, inner.SizeInBytes);
        Assert.Equal(4u, inner.Alignment);
        Assert.Equal(16u, outer.SizeInBytes);
        Assert.Equal(4u, outer.Alignment);
        Assert.Equal("Items", section.Strings[(int)items.NameId]);
        Assert.Equal(0u, items.Offset);
        Assert.Equal(12u, items.SizeInBytes);
        Assert.Equal(5, itemsType.Kind);
        Assert.Equal(3u, itemsType.B);
        Assert.Equal(4, itemElementType.Kind);
    }

    private static FeatherIrModule ReadGeneratedIr(string source)
    {
        return FeatherIr.Read(ExtractGeneratedIrBytes(source));
    }

    private static byte[] ExtractGeneratedIrBytes(string source)
    {
        var irInitializer = source.Split("new byte[] { ", StringSplitOptions.None)[1].Split(" };", StringSplitOptions.None)[0];
        return irInitializer.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(value => Convert.ToByte(value[2..], 16))
            .ToArray();
    }

    private static byte[] ExtractTypedIrBytes(string source)
    {
        using var stream = new MemoryStream(ExtractGeneratedIrBytes(source));
        using var reader = new BinaryReader(stream);
        Assert.Equal("FEIR", new string(reader.ReadChars(4)));
        _ = reader.ReadUInt16();
        _ = reader.ReadUInt16();
        Assert.Equal(1, reader.ReadByte());
        _ = reader.ReadByte();
        var sectionCount = reader.ReadUInt16();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadUInt32();
        var resourceCount = reader.ReadUInt32();
        _ = reader.ReadUInt32();
        var instructionCount = reader.ReadUInt32();
        _ = reader.ReadUInt32();

        stream.Position += resourceCount * 15;
        stream.Position += instructionCount * 8;

        var sections = new List<(uint Kind, uint ByteLength)>();
        for (var i = 0; i < sectionCount; i++)
        {
            sections.Add((reader.ReadUInt32(), reader.ReadUInt32()));
        }

        foreach (var (kind, byteLength) in sections)
        {
            var payload = reader.ReadBytes(checked((int)byteLength));
            if (kind == ShaderIrModuleWriter.SectionKind)
            {
                return payload;
            }
        }

        throw new InvalidDataException("Generated IR did not contain typed shader IR section 7.");
    }

    private static SyntaxTree GetGeneratedTree(Compilation compilation, IEnumerable<Diagnostic> generatorDiagnostics, string fileName)
    {
        var matches = compilation.SyntaxTrees
            .Where(tree => tree.FilePath.EndsWith(fileName, StringComparison.Ordinal))
            .ToArray();
        if (matches.Length == 1)
        {
            return matches[0];
        }

        var diagnostics = generatorDiagnostics
            .Concat(compilation.GetDiagnostics())
            .Select(diagnostic => diagnostic.ToString());
        throw new InvalidOperationException($"Expected one generated tree ending with '{fileName}', found {matches.Length}.\n" + string.Join("\n", diagnostics));
    }

    private static int IndexOf(FeatherIrInstructionOpcode[] opcodes, FeatherIrInstructionOpcode opcode)
        => Array.IndexOf(opcodes, opcode);

    private static ShaderModuleModel LowerTypedModule(string source)
    {
        var compilation = CreateCompilation(source);
        var syntaxTree = compilation.SyntaxTrees.Single();
        var syntax = syntaxTree.GetRoot().DescendantNodes().OfType<StructDeclarationSyntax>().Single(static s =>
            s.AttributeLists.SelectMany(static list => list.Attributes).Any(static attribute =>
                attribute.Name.ToString() is "Kernel" or "KernelAttribute"));
        var semanticModel = compilation.GetSemanticModel(syntaxTree);
        var symbol = semanticModel.GetDeclaredSymbol(syntax)!;
        var model = ShaderModelFactory.Create(syntax, symbol)!;
        return ShaderIrLowerer.Lower(model, semanticModel, CancellationToken.None)!;
    }

    private static void AssertLocal(ShaderStatement statement, string name)
    {
        var local = Assert.IsType<ShaderLocalDeclarationStatement>(statement);
        Assert.Equal(name, local.VariableName);
    }

    private static TypedIrSection ReadTypedIrSection(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);
        var magic = System.Text.Encoding.ASCII.GetString(reader.ReadBytes(4));
        var major = reader.ReadUInt16();
        var minor = reader.ReadUInt16();
        var endian = reader.ReadByte();
        reader.ReadByte();
        var headerSize = reader.ReadUInt16();
        var entryId = reader.ReadUInt32();
        var functionRange = ReadRange(reader);
        var typeRange = ReadRange(reader);
        var structRange = ReadRange(reader);
        var structFieldRange = ReadRange(reader);
        var statementRange = ReadRange(reader);
        var expressionRange = ReadRange(reader);
        var lvalueRange = ReadRange(reader);
        var childRange = ReadRange(reader);
        var argumentRange = ReadRange(reader);
        var parameterRange = ReadRange(reader);
        var stringOffset = reader.ReadUInt32();
        var stringLength = reader.ReadUInt32();

        Assert.Equal("FTIR", magic);
        Assert.Equal(1, endian);
        Assert.Equal(104, headerSize);

        var structFieldRecordSize = minor >= 1 ? 20u : 16u;
        var tableRanges = new[]
        {
            (functionRange.Offset, functionRange.Count, 25u),
            (typeRange.Offset, typeRange.Count, 17u),
            (structRange.Offset, structRange.Count, 24u),
            (structFieldRange.Offset, structFieldRange.Count, structFieldRecordSize),
            (statementRange.Offset, statementRange.Count, 29u),
            (expressionRange.Offset, expressionRange.Count, 33u),
            (lvalueRange.Offset, lvalueRange.Count, 21u),
            (childRange.Offset, childRange.Count, 4u),
            (argumentRange.Offset, argumentRange.Count, 4u),
            (parameterRange.Offset, parameterRange.Count, 9u),
        };

        var previousEnd = (uint)headerSize;
        foreach (var (offset, count, recordSize) in tableRanges)
        {
            Assert.InRange(offset, previousEnd, (uint)data.Length);
            var end = checked(offset + (count * recordSize));
            Assert.InRange(end, offset, (uint)data.Length);
            previousEnd = end;
        }

        Assert.Equal(previousEnd, stringOffset);
        Assert.Equal(data.Length, checked((int)(stringOffset + stringLength)));

        var functions = ReadRecords(functionRange, 25, () => new TypedIrFunction(
            reader.ReadByte(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32()));

        var types = ReadRecords(typeRange, 17, () => new TypedIrType(
            reader.ReadByte(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32()));

        var structs = ReadRecords(structRange, 24, () => new TypedIrStruct(
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32()));

        var structFields = ReadRecords(structFieldRange, (int)structFieldRecordSize, () =>
        {
            var field = new TypedIrStructField(
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                reader.ReadUInt32(),
                minor >= 1 ? reader.ReadUInt32() : 0u);
            return field;
        });

        var statements = ReadRecords(statementRange, 29, () => new TypedIrStatement(
            reader.ReadByte(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32()));

        var expressions = ReadRecords(expressionRange, 33, () => new TypedIrExpression(
            reader.ReadByte(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32()));

        var lvalues = ReadRecords(lvalueRange, 21, () => new TypedIrLValue(
            reader.ReadByte(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32()));

        var children = ReadRecords(childRange, 4, reader.ReadUInt32);
        var arguments = ReadRecords(argumentRange, 4, reader.ReadUInt32);
        var parameters = ReadRecords(parameterRange, 9, () => new TypedIrParameter(
            reader.ReadByte(),
            reader.ReadUInt32(),
            reader.ReadUInt32()));
        var strings = ReadStrings(stringOffset, stringLength);

        Assert.InRange(entryId, 0u, (uint)functions.Count - 1);
        ValidateStatementReferences(statements, expressions.Count, lvalues.Count, children.Count);
        ValidateExpressionReferences(expressions, arguments.Count);
        ValidateFunctionReferences(functions, statements.Count, parameters.Count);

        return new TypedIrSection(
            magic,
            major,
            minor,
            headerSize,
            entryId,
            functions,
            types,
            structs,
            structFields,
            statements,
            expressions,
            lvalues,
            children,
            arguments,
            parameters,
            strings);

        static TypedIrRange ReadRange(BinaryReader reader)
            => new(reader.ReadUInt32(), reader.ReadUInt32());

        List<T> ReadRecords<T>(TypedIrRange range, int recordSize, Func<T> read)
        {
            stream.Position = range.Offset;
            var records = new List<T>((int)range.Count);
            for (var i = 0; i < range.Count; i++)
            {
                var start = stream.Position;
                records.Add(read());
                Assert.Equal(start + recordSize, stream.Position);
            }

            return records;
        }

        List<string> ReadStrings(uint offset, uint byteLength)
        {
            stream.Position = offset;
            var end = checked(offset + byteLength);
            var count = reader.ReadUInt32();
            var values = new List<string>((int)count);
            for (var i = 0; i < count; i++)
            {
                var length = reader.ReadUInt32();
                var bytes = reader.ReadBytes(checked((int)length));
                Assert.Equal(length, (uint)bytes.Length);
                values.Add(System.Text.Encoding.UTF8.GetString(bytes));
            }

            Assert.Equal(end, (uint)stream.Position);
            return values;
        }
    }

    private static void ValidateStatementReferences(
        IReadOnlyList<TypedIrStatement> statements,
        int expressionCount,
        int lvalueCount,
        int childCount)
    {
        foreach (var statement in statements)
        {
            Assert.True(statement.Kind is >= 1 and <= 15, $"Unexpected statement kind {statement.Kind}.");
            switch (statement.Kind)
            {
                case 1:
                    if (statement.ChildCount > 0)
                    {
                        Assert.InRange(statement.FirstChild, 0u, (uint)childCount - statement.ChildCount);
                    }
                    else
                    {
                        Assert.Equal(uint.MaxValue, statement.FirstChild);
                    }
                    break;
                case 2:
                    Assert.True(statement.A == uint.MaxValue || statement.A < expressionCount);
                    break;
                case 3:
                case 4:
                case 14:
                    Assert.InRange(statement.A, 0u, (uint)lvalueCount - 1);
                    break;
                case 5:
                    Assert.InRange(statement.A, 0u, (uint)expressionCount - 1);
                    Assert.InRange(statement.B, 0u, (uint)statements.Count - 1);
                    Assert.True(statement.C == uint.MaxValue || statement.C < statements.Count);
                    break;
                case 7:
                case 8:
                    Assert.True(statement.A < statements.Count || statement.A < expressionCount);
                    Assert.InRange(statement.B, 0u, (uint)System.Math.Max(expressionCount, statements.Count) - 1);
                    break;
                case 11:
                case 12:
                    Assert.True(statement.A == uint.MaxValue || statement.A < expressionCount);
                    break;
                case 15:
                    Assert.NotEqual(0u, statement.A);
                    break;
            }
        }
    }

    private static void ValidateExpressionReferences(IReadOnlyList<TypedIrExpression> expressions, int argumentCount)
    {
        foreach (var expression in expressions)
        {
            Assert.True(expression.Kind is >= 1 and <= 23, $"Unexpected expression kind {expression.Kind}.");
            if (expression.ArgumentCount > 0)
            {
                Assert.InRange(expression.FirstArgument, 0u, (uint)argumentCount - expression.ArgumentCount);
            }
            else
            {
                Assert.Equal(uint.MaxValue, expression.FirstArgument);
            }
        }
    }

    private static void ValidateFunctionReferences(
        IReadOnlyList<TypedIrFunction> functions,
        int statementCount,
        int parameterCount)
    {
        foreach (var function in functions)
        {
            Assert.InRange(function.BodyStatementId, 0u, (uint)statementCount - 1);
            if (function.ParameterCount > 0)
            {
                Assert.InRange(function.FirstParameter, 0u, (uint)parameterCount - function.ParameterCount);
            }
            else
            {
                Assert.Equal(uint.MaxValue, function.FirstParameter);
            }
        }
    }

    private static CSharpCompilation CreateCompilation(string source)
    {
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(assembly => !assembly.IsDynamic && !string.IsNullOrWhiteSpace(assembly.Location))
            .Select(assembly => MetadataReference.CreateFromFile(assembly.Location))
            .Concat([MetadataReference.CreateFromFile(typeof(KernelAttribute).Assembly.Location)])
            .Distinct(MetadataReferenceComparer.Instance);

        return CSharpCompilation.Create(
            "Scratch",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Enable));
    }

    private static async Task<ImmutableArray<Diagnostic>> AnalyzeAsync(string source)
    {
        var analyzer = new FeatherAnalyzer();
        return await CreateCompilation(source)
            .WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(analyzer))
            .GetAnalyzerDiagnosticsAsync();
    }

    private sealed class MetadataReferenceComparer : IEqualityComparer<MetadataReference>
    {
        public static readonly MetadataReferenceComparer Instance = new();

        public bool Equals(MetadataReference? x, MetadataReference? y)
            => StringComparer.OrdinalIgnoreCase.Equals(x?.Display, y?.Display);

        public int GetHashCode(MetadataReference obj)
            => StringComparer.OrdinalIgnoreCase.GetHashCode(obj.Display ?? string.Empty);
    }

    private sealed record TypedIrSection(
        string Magic,
        ushort MajorVersion,
        ushort MinorVersion,
        ushort HeaderSize,
        uint EntryFunctionId,
        IReadOnlyList<TypedIrFunction> Functions,
        IReadOnlyList<TypedIrType> Types,
        IReadOnlyList<TypedIrStruct> Structs,
        IReadOnlyList<TypedIrStructField> StructFields,
        IReadOnlyList<TypedIrStatement> Statements,
        IReadOnlyList<TypedIrExpression> Expressions,
        IReadOnlyList<TypedIrLValue> LValues,
        IReadOnlyList<uint> Children,
        IReadOnlyList<uint> Arguments,
        IReadOnlyList<TypedIrParameter> Parameters,
        IReadOnlyList<string> Strings);

    private readonly record struct TypedIrRange(uint Offset, uint Count);

    private readonly record struct TypedIrFunction(
        byte Kind,
        uint NameId,
        uint MangledNameId,
        uint ReturnTypeId,
        uint FirstParameter,
        uint ParameterCount,
        uint BodyStatementId);

    private readonly record struct TypedIrType(byte Kind, uint A, uint B, uint C, uint D);

    private readonly record struct TypedIrStruct(
        uint NameId,
        uint FullyQualifiedNameId,
        uint FirstField,
        uint FieldCount,
        uint SizeInBytes,
        uint Alignment);

    private readonly record struct TypedIrStructField(uint NameId, uint TypeId, uint Offset, uint SizeInBytes, uint Flags);

    private readonly record struct TypedIrStatement(byte Kind, uint A, uint B, uint C, uint Op, uint NameId, uint FirstChild, uint ChildCount);

    private readonly record struct TypedIrExpression(byte Kind, uint TyId, uint A, uint B, uint C, uint NameId, uint Op, uint FirstArgument, uint ArgumentCount);

    private readonly record struct TypedIrLValue(byte Kind, uint TyId, uint A, uint B, uint C, uint NameId);

    private readonly record struct TypedIrParameter(byte Direction, uint NameId, uint TypeId);
}
