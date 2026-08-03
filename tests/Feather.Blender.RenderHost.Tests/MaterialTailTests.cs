using System.Text.Json;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

public sealed class MaterialTailTests
{
    private const string NodeGuid = "8a2bf9cd-65a5-46ff-9c26-d2a9b4ca8746";

    [Fact]
    public void ProceduralBumpReevaluatesHeightGraphAtFourUvOffsets()
    {
        var nodes = BaseConstants();
        nodes.Insert(0, Node("uv", "uv_map", "UV"));
        nodes.Add(Node(
            "noise", "noise_texture", "Fac",
            Inputs(
                ("Vector", "uv"), ("Scale", "c0"), ("Detail", "c0"),
                ("Roughness", "c0"), ("Lacunarity", "c0"), ("Distortion", "c0")),
            new { noise_dimensions = "3D", normalize = false }));
        nodes.Add(Node(
            "bump", "bump", "Normal",
            Inputs(("Height", "noise"), ("Strength", "c1"), ("Distance", "c1"), ("Normal", "c0")),
            new { invert = false }, NodeGuid));

        var expression = Compile(nodes, normal: "bump");

        Assert.Contains(
            expression.Instructions.ToArray(),
            item => item.Op == (int)SceneMaterialExpressionOp.BumpEvaluated);
        Assert.Equal(4, expression.Instructions.ToArray().Count(
            item => item.Op == (int)SceneMaterialExpressionOp.Noise) - 1);
        Assert.True(expression.Instructions.Length < SceneMaterialExpression.MaxInstructions);
        Console.WriteLine(
            $"FEATHER_T4_BUMP_LOWERING source={nodes.Count} lowered={expression.Instructions.Length} " +
            $"vmLimit={SceneMaterialExpression.MaxInstructions}");
    }

    [Theory]
    [InlineData("F2", "MANHATTAN", 1, 1)]
    [InlineData("F1", "CHEBYCHEV", 0, 2)]
    [InlineData("F2", "MINKOWSKI", 1, 3)]
    public void VoronoiVariantsAreLoweredWithoutSilentF1Fallback(
        string feature, string distance, int featureCode, int distanceCode)
    {
        var nodes = BaseConstants();
        nodes.Insert(0, Node("uv", "uv_map", "UV"));
        nodes.Add(Node(
            "voronoi", "voronoi_texture", "Distance",
            Inputs(("Vector", "uv"), ("Scale", "c1"), ("Randomness", "c1"), ("Exponent", "c1")),
            new { feature, distance, voronoi_dimensions = "3D", normalize = false }, NodeGuid));

        var instruction = Compile(nodes, baseColor: "voronoi").Instructions.Span[^1];

        Assert.Equal((int)SceneMaterialExpressionOp.Voronoi, instruction.Op);
        Assert.Equal(featureCode, instruction.Parameters.Y);
        Assert.Equal(distanceCode, instruction.Parameters.Z);
    }

    [Theory]
    [InlineData("LINEAR", 0)]
    [InlineData("SMOOTHSTEP", 1)]
    [InlineData("SMOOTHERSTEP", 2)]
    public void NonLinearMapRangeModesHaveDistinctVmCodes(string mode, int expected)
    {
        var nodes = BaseConstants();
        nodes.Add(Node(
            "map", "map_range", "Result",
            Inputs(
                ("Value", "c1"), ("From Min", "c0"), ("From Max", "c1"),
                ("To Min", "c0"), ("To Max", "c1")),
            new { data_type = "FLOAT", interpolation_type = mode, clamp = true }, NodeGuid));

        var instruction = Compile(nodes, baseColor: "map").Instructions.Span[^1];

        Assert.Equal(expected, instruction.Parameters.Y);
    }

    [Fact]
    public void UnsupportedHsvRampCarriesTheOriginatingNodeGuid()
    {
        var nodes = BaseConstants();
        nodes.Add(Node(
            "ramp", "color_ramp", "Color",
            Inputs(("Factor", "c1")),
            new
            {
                interpolation = "LINEAR",
                color_mode = "HSV",
                hue_interpolation = "NEAR",
                elements = new[]
                {
                    new { position = 0.0f, color = new[] { 0.0f, 0.0f, 0.0f, 1.0f } },
                    new { position = 1.0f, color = new[] { 1.0f, 1.0f, 1.0f, 1.0f } }
                }
            },
            NodeGuid));

        var exception = Assert.Throws<MaterialExpressionException>(
            () => Compile(nodes, baseColor: "ramp"));

        Assert.Equal(NodeGuid, exception.NodeGuid);
        Assert.Equal("MATERIAL_EXPRESSION_UNSUPPORTED", exception.ErrorCode);
        Assert.Contains("HSV", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BumpExpansionOverVmLimitFailsLoudlyAtTheBumpNode()
    {
        var nodes = BaseConstants();
        nodes.Insert(0, Node("uv", "uv_map", "UV"));
        nodes.Add(Node(
            "noise", "noise_texture", "Fac",
            Inputs(
                ("Vector", "uv"), ("Scale", "c0"), ("Detail", "c0"),
                ("Roughness", "c0"), ("Lacunarity", "c0"), ("Distortion", "c0")),
            new { noise_dimensions = "3D", normalize = false }));
        var previous = "noise";
        for (var index = 0; index < 26; index++)
        {
            var id = $"math{index}";
            nodes.Add(Node(
                id,
                "math",
                "Value",
                Inputs(("Value", previous)),
                new { operation = "ABSOLUTE", use_clamp = false }));
            previous = id;
        }
        nodes.Add(Node(
            "bump", "bump", "Normal",
            Inputs(("Height", previous), ("Strength", "c1"), ("Distance", "c1"), ("Normal", "c0")),
            new { invert = false }, NodeGuid));

        var exception = Assert.Throws<MaterialExpressionException>(
            () => Compile(nodes, normal: "bump"));

        Assert.Equal("MATERIAL_INSTRUCTION_LIMIT", exception.ErrorCode);
        Assert.Equal(NodeGuid, exception.NodeGuid);
        Assert.Contains("expanded", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StructuredFaultInjectionUsesStableCodesAndCompleteFields()
    {
        var cases = new[]
        {
            RenderHostDiagnostic.FromException(
                new InvalidOperationException("shader compilation failed: injected")),
            RenderHostDiagnostic.ForMaterial(
                new SceneMaterialMetadata
                {
                    MaterialId = "material-7", Name = "Injected Unsupported", NodeTree = "Nodes"
                },
                "UNSUPPORTED_MATERIAL_NODE: injected",
                NodeGuid),
            RenderHostDiagnostic.FromException(new OutOfMemoryException("injected OOM"))
        };

        Assert.Equal(
            ["SHADER_COMPILE_ERROR", "UNSUPPORTED_MATERIAL_NODE", "HOST_OUT_OF_MEMORY"],
            cases.Select(item => item.ErrorCode));
        foreach (var diagnostic in cases)
        {
            var line = JsonSerializer.Serialize(
                new { @event = "diagnostic", value = diagnostic }, ProtocolJson.Options);
            Console.WriteLine(line);
            Assert.Contains("\"errorCode\"", line, StringComparison.Ordinal);
            Assert.Contains("\"severity\"", line, StringComparison.Ordinal);
            Assert.Contains("\"passGuid\"", line, StringComparison.Ordinal);
            Assert.Contains("\"nodeGuid\"", line, StringComparison.Ordinal);
            Assert.Contains("\"sourcePath\"", line, StringComparison.Ordinal);
            Assert.Contains("\"action\"", line, StringComparison.Ordinal);
            Assert.Contains("\"context\"", line, StringComparison.Ordinal);
        }
    }

    private static SceneMaterialExpression Compile(
        List<Dictionary<string, object?>> nodes,
        string baseColor = "c0",
        string normal = "c0")
    {
        var outputs = new Dictionary<string, string>
        {
            ["baseColor"] = baseColor,
            ["metallic"] = "c0",
            ["roughness"] = "c1",
            ["ior"] = "c1",
            ["diffuseRoughness"] = "c0",
            ["transmissionWeight"] = "c0",
            ["sheenWeight"] = "c0",
            ["sheenColor"] = "c1",
            ["clearcoatWeight"] = "c0",
            ["clearcoatRoughness"] = "c0",
            ["emissionColor"] = "c0",
            ["emissionStrength"] = "c0",
            ["alpha"] = "c1",
            ["normal"] = normal
        };
        var json = JsonSerializer.Serialize(new
        {
            version = 1,
            hash = new string('0', 64),
            nodes,
            outputs,
            textures = Array.Empty<string>()
        }, ProtocolJson.Options);
        using var document = JsonDocument.Parse(json);
        return MaterialExpressionCompiler.Compile(
            document.RootElement.Clone(), new Dictionary<string, int>())!;
    }

    private static List<Dictionary<string, object?>> BaseConstants() =>
    [
        new Dictionary<string, object?> { ["id"] = "c0", ["op"] = "constant", ["value"] = 0.0f },
        new Dictionary<string, object?> { ["id"] = "c1", ["op"] = "constant", ["value"] = 1.0f }
    ];

    private static Dictionary<string, string> Inputs(params (string Name, string Id)[] values)
        => values.ToDictionary(item => item.Name, item => item.Id, StringComparer.Ordinal);

    private static Dictionary<string, object?> Node(
        string id,
        string op,
        string output,
        Dictionary<string, string>? inputs = null,
        object? parameters = null,
        string nodeGuid = "")
        => new()
        {
            ["id"] = id,
            ["op"] = op,
            ["output"] = output,
            ["nodeGuid"] = nodeGuid,
            ["inputs"] = inputs ?? new Dictionary<string, string>(),
            ["params"] = parameters ?? new { }
        };
}
