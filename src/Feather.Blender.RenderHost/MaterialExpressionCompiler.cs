using System.Text.Json;
using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost;

/// <summary>Validates and lowers the snapshot expression DAG to the bounded raster shader VM.</summary>
internal static class MaterialExpressionCompiler
{
    public static SceneMaterialExpression? Compile(
        JsonElement? expression,
        IReadOnlyDictionary<string, int> textureIndices)
    {
        if (expression is null || expression.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return null;
        }

        var root = expression.Value;
        Require(root.ValueKind == JsonValueKind.Object, "materialExpression must be an object");
        Require(ReadInt(root, "version") == 1, "materialExpression version must be 1");
        var hash = ReadString(root, "hash");
        Require(hash.Length == 64, "materialExpression hash must contain 64 characters");
        var nodes = Required(root, "nodes");
        Require(nodes.ValueKind == JsonValueKind.Array, "materialExpression nodes must be an array");
        Require(nodes.GetArrayLength() is > 0 and <= SceneMaterialExpression.MaxInstructions,
            $"materialExpression must contain 1-{SceneMaterialExpression.MaxInstructions} nodes");

        var indices = new Dictionary<string, int>(nodes.GetArrayLength(), StringComparer.Ordinal);
        var instructions = new SceneMaterialExpressionInstruction[nodes.GetArrayLength()];
        var parameters = new List<float4>();
        var textureIndex = SceneMaterial.NoTexture;
        var nodeIndex = 0;
        foreach (var node in nodes.EnumerateArray())
        {
            Require(node.ValueKind == JsonValueKind.Object, "materialExpression node must be an object");
            var id = ReadString(node, "id");
            Require(indices.TryAdd(id, nodeIndex), $"materialExpression contains duplicate node ID '{id}'");
            instructions[nodeIndex] = CompileNode(
                node,
                indices,
                parameters,
                textureIndices,
                ref textureIndex);
            nodeIndex++;
        }

        var outputs = Required(root, "outputs");
        Require(outputs.ValueKind == JsonValueKind.Object, "materialExpression outputs must be an object");
        int Output(string name) => ResolveReference(ReadString(outputs, name), indices, name);
        return new SceneMaterialExpression(
            hash,
            instructions,
            parameters.ToArray(),
            new SceneMaterialExpressionOutputs(
                Output("baseColor"),
                Output("metallic"),
                Output("roughness"),
                Output("ior"),
                Output("diffuseRoughness"),
                Output("transmissionWeight"),
                Output("sheenWeight"),
                Output("sheenColor"),
                Output("clearcoatWeight"),
                Output("clearcoatRoughness"),
                Output("emissionColor"),
                Output("emissionStrength"),
                Output("alpha"),
                Output("normal")),
            textureIndex);
    }

    private static SceneMaterialExpressionInstruction CompileNode(
        JsonElement node,
        IReadOnlyDictionary<string, int> indices,
        List<float4> parameters,
        IReadOnlyDictionary<string, int> textureIndices,
        ref int textureIndex)
    {
        var opName = ReadString(node, "op");
        var instruction = EmptyInstruction();
        if (opName == "constant")
        {
            instruction.Op = (int)SceneMaterialExpressionOp.Constant;
            instruction.Value = ReadValue(Required(node, "value"), "constant value");
            return instruction;
        }

        var inputs = Required(node, "inputs");
        var nodeParameters = Required(node, "params");
        Require(inputs.ValueKind == JsonValueKind.Object, $"{opName} inputs must be an object");
        Require(nodeParameters.ValueKind == JsonValueKind.Object, $"{opName} params must be an object");
        var output = ReadString(node, "output");
        int Input(params string[] names)
        {
            foreach (var name in names)
            {
                if (inputs.TryGetProperty(name, out var value))
                {
                    return ResolveReference(value.GetString() ?? "", indices, $"{opName}.{name}");
                }
            }
            throw Error($"{opName} is missing input '{names[0]}'");
        }

        switch (opName)
        {
            case "rgb":
                instruction.Op = (int)SceneMaterialExpressionOp.Constant;
                instruction.Value = ReadValue(Required(nodeParameters, "value"), "RGB value");
                break;
            case "value":
                instruction.Op = (int)SceneMaterialExpressionOp.Constant;
                instruction.Value = Broadcast(ReadFloat(nodeParameters, "value"));
                break;
            case "texture_coordinate":
            case "uv_map":
                instruction.Op = (int)SceneMaterialExpressionOp.Uv;
                break;
            case "image_texture":
            {
                instruction.Op = (int)(output == "Alpha"
                    ? SceneMaterialExpressionOp.ImageAlpha
                    : SceneMaterialExpressionOp.ImageColor);
                Require(output is "Color" or "Alpha", $"image_texture output '{output}' is unsupported");
                instruction.A = Input("Vector");
                var id = ReadString(nodeParameters, "textureId");
                Require(textureIndices.TryGetValue(id, out var resolvedTexture),
                    $"image_texture references missing texture '{id}'");
                Require(textureIndex is SceneMaterial.NoTexture || textureIndex == resolvedTexture,
                    "one raster material expression cannot bind more than one image texture yet");
                textureIndex = resolvedTexture;
                break;
            }
            case "noise_texture":
                instruction.Op = (int)SceneMaterialExpressionOp.Noise;
                instruction.A = Input("Vector");
                instruction.B = Input("Scale");
                instruction.C = Input("Detail");
                instruction.D = Input("Roughness");
                instruction.E = Input("Lacunarity");
                instruction.F = Input("Distortion");
                instruction.Parameters = new float4(output == "Color" ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
                Require(ReadOptionalString(nodeParameters, "noise_dimensions", "3D") is "2D" or "3D",
                    "only 2D and 3D Noise Texture modes are supported by raster evaluation");
                break;
            case "voronoi_texture":
                instruction.Op = (int)SceneMaterialExpressionOp.Voronoi;
                instruction.A = Input("Vector");
                instruction.B = Input("Scale");
                instruction.C = Input("Randomness");
                instruction.Parameters = new float4(output == "Color" ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
                Require(ReadOptionalString(nodeParameters, "feature", "F1") == "F1",
                    "only Voronoi F1 is supported by raster evaluation");
                Require(ReadOptionalString(nodeParameters, "distance", "EUCLIDEAN") == "EUCLIDEAN",
                    "only Euclidean Voronoi distance is supported by raster evaluation");
                break;
            case "gradient_texture":
                instruction.Op = (int)SceneMaterialExpressionOp.Gradient;
                instruction.A = Input("Vector");
                instruction.Parameters = new float4(GradientCode(ReadString(nodeParameters, "gradient_type")), 0.0f, 0.0f, 0.0f);
                break;
            case "checker_texture":
                instruction.Op = (int)SceneMaterialExpressionOp.Checker;
                instruction.A = Input("Vector");
                instruction.B = Input("Color1");
                instruction.C = Input("Color2");
                instruction.D = Input("Scale");
                instruction.Parameters = new float4(output == "Fac" ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
                break;
            case "mix":
                instruction.Op = (int)SceneMaterialExpressionOp.Mix;
                instruction.A = Input("Factor", "Factor_Vector");
                instruction.B = Input("A");
                instruction.C = Input("B");
                instruction.Parameters = new float4(
                    ReadOptionalBool(nodeParameters, "clamp_factor", true) ? 1.0f : 0.0f,
                    ReadOptionalBool(nodeParameters, "clamp_result", false) ? 1.0f : 0.0f,
                    0.0f,
                    0.0f);
                Require(ReadOptionalString(nodeParameters, "blend_type", "MIX") == "MIX",
                    "only MIX mode is supported for the generic Mix node");
                break;
            case "color_ramp":
                instruction.Op = (int)SceneMaterialExpressionOp.ColorRamp;
                instruction.A = Input("Factor");
                instruction.ParameterOffset = parameters.Count;
                var rampInterpolation = RampInterpolationCode(ReadString(nodeParameters, "interpolation"));
                var elements = Required(nodeParameters, "elements");
                Require(elements.ValueKind == JsonValueKind.Array && elements.GetArrayLength() >= 2,
                    "ColorRamp must contain at least two elements");
                foreach (var element in elements.EnumerateArray())
                {
                    parameters.Add(ReadValue(Required(element, "color"), "ColorRamp color"));
                    parameters.Add(new float4(ReadFloat(element, "position"), 0.0f, 0.0f, 0.0f));
                    instruction.ParameterCount++;
                }
                instruction.Parameters = new float4(
                    rampInterpolation, output == "Alpha" ? 1.0f : 0.0f, 0.0f, 0.0f);
                break;
            case "rgb_curves":
                instruction.Op = (int)SceneMaterialExpressionOp.RgbCurves;
                instruction.A = Input("Factor");
                instruction.B = Input("Color");
                instruction.ParameterOffset = parameters.Count;
                var curves = Required(nodeParameters, "curves");
                Require(curves.ValueKind == JsonValueKind.Array && curves.GetArrayLength() == 4,
                    "RGB Curves must contain four curves");
                var curveIndex = 0;
                foreach (var curve in curves.EnumerateArray())
                {
                    Require(curve.ValueKind == JsonValueKind.Array && curve.GetArrayLength() >= 2,
                        "each RGB curve must contain at least two points");
                    foreach (var point in curve.EnumerateArray())
                    {
                        Require(point.ValueKind == JsonValueKind.Array && point.GetArrayLength() == 2,
                            "RGB curve points must contain x and y");
                        parameters.Add(new float4(
                            point[0].GetSingle(), point[1].GetSingle(), curveIndex, 0.0f));
                        instruction.ParameterCount++;
                    }
                    curveIndex++;
                }
                break;
            case "math":
                instruction.Op = (int)SceneMaterialExpressionOp.Math;
                instruction.A = Input("Value");
                instruction.B = Input("Value_001");
                instruction.C = inputs.TryGetProperty("Value_002", out _) ? Input("Value_002") : instruction.A;
                instruction.Parameters = new float4(
                    MathCode(ReadString(nodeParameters, "operation")),
                    ReadOptionalBool(nodeParameters, "use_clamp", false) ? 1.0f : 0.0f,
                    0.0f,
                    0.0f);
                break;
            case "vector_math":
                instruction.Op = (int)SceneMaterialExpressionOp.VectorMath;
                instruction.A = Input("Vector");
                instruction.B = inputs.TryGetProperty("Vector_001", out _) ? Input("Vector_001") : instruction.A;
                instruction.C = inputs.TryGetProperty("Vector_002", out _) ? Input("Vector_002") : instruction.A;
                instruction.D = inputs.TryGetProperty("Scale", out _) ? Input("Scale") : instruction.A;
                instruction.Parameters = new float4(
                    VectorMathCode(ReadString(nodeParameters, "operation")),
                    output == "Value" ? 1.0f : 0.0f,
                    0.0f,
                    0.0f);
                break;
            case "map_range":
                instruction.Op = (int)SceneMaterialExpressionOp.MapRange;
                instruction.A = Input("Value");
                instruction.B = Input("From Min");
                instruction.C = Input("From Max");
                instruction.D = Input("To Min");
                instruction.E = Input("To Max");
                Require(ReadOptionalString(nodeParameters, "data_type", "FLOAT") == "FLOAT" &&
                        ReadOptionalString(nodeParameters, "interpolation_type", "LINEAR") == "LINEAR",
                    "only linear float Map Range is supported by raster evaluation");
                instruction.Parameters = new float4(
                    ReadOptionalBool(nodeParameters, "clamp", true) ? 1.0f : 0.0f,
                    0.0f,
                    0.0f,
                    0.0f);
                break;
            case "mix_rgb":
                instruction.Op = (int)SceneMaterialExpressionOp.MixRgb;
                instruction.A = Input("Factor");
                instruction.B = Input("Color1");
                instruction.C = Input("Color2");
                instruction.Parameters = new float4(
                    MixRgbCode(ReadString(nodeParameters, "blend_type")),
                    ReadOptionalBool(nodeParameters, "use_clamp", false) ? 1.0f : 0.0f,
                    0.0f,
                    0.0f);
                break;
            case "hue_saturation_value":
                instruction.Op = (int)SceneMaterialExpressionOp.HueSaturationValue;
                instruction.A = Input("Color");
                instruction.B = Input("Factor");
                instruction.C = Input("Hue");
                instruction.D = Input("Saturation");
                instruction.E = Input("Value");
                break;
            case "mapping":
                instruction.Op = (int)SceneMaterialExpressionOp.Mapping;
                instruction.A = Input("Vector");
                instruction.B = Input("Location");
                instruction.C = Input("Rotation");
                instruction.D = Input("Scale");
                instruction.Parameters = new float4(MappingCode(ReadString(nodeParameters, "vector_type")), 0.0f, 0.0f, 0.0f);
                break;
            case "normal_map":
                instruction.Op = (int)SceneMaterialExpressionOp.NormalMap;
                instruction.A = Input("Color");
                instruction.B = Input("Strength");
                break;
            case "bump":
                throw Error("Bump evaluation requires derivative re-evaluation and is not in the M1 raster VM");
            case "separate_xyz":
                instruction.Op = (int)SceneMaterialExpressionOp.SeparateXyz;
                instruction.A = Input("Vector");
                instruction.Parameters = new float4(
                    output switch { "X" => 0.0f, "Y" => 1.0f, "Z" => 2.0f, _ => throw Error($"unsupported Separate XYZ output '{output}'") },
                    0.0f,
                    0.0f,
                    0.0f);
                break;
            case "combine_xyz":
                instruction.Op = (int)SceneMaterialExpressionOp.CombineXyz;
                instruction.A = Input("X");
                instruction.B = Input("Y");
                instruction.C = Input("Z");
                break;
            case "fresnel":
                instruction.Op = (int)SceneMaterialExpressionOp.Fresnel;
                instruction.A = Input("IOR");
                instruction.B = Input("Normal");
                break;
            case "layer_weight":
                instruction.Op = (int)SceneMaterialExpressionOp.LayerWeight;
                instruction.A = Input("Blend");
                instruction.B = Input("Normal");
                instruction.Parameters = new float4(output == "Facing" ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
                break;
            case "mix_shader":
                instruction.Op = (int)SceneMaterialExpressionOp.MixShader;
                instruction.A = Input("Factor");
                instruction.B = Input("A");
                instruction.C = Input("B");
                break;
            case "add_shader":
                instruction.Op = (int)SceneMaterialExpressionOp.AddShader;
                instruction.A = Input("A");
                instruction.B = Input("B");
                break;
            default:
                throw Error($"material expression op '{opName}' is unsupported by raster evaluation");
        }
        return instruction;
    }

    private static SceneMaterialExpressionInstruction EmptyInstruction() => new()
    {
        Value = float4.Zero,
        Parameters = float4.Zero,
        A = -1,
        B = -1,
        C = -1,
        D = -1,
        E = -1,
        F = -1,
        G = -1,
        H = -1
    };

    private static int ResolveReference(string id, IReadOnlyDictionary<string, int> indices, string label)
    {
        Require(indices.TryGetValue(id, out var index), $"{label} references unknown or forward node '{id}'");
        return index;
    }

    private static float4 ReadValue(JsonElement value, string label)
    {
        if (value.ValueKind == JsonValueKind.Number)
        {
            return Broadcast(value.GetSingle());
        }
        Require(value.ValueKind == JsonValueKind.Array, $"{label} must be numeric or an array");
        var components = value.EnumerateArray().Select(item => item.GetSingle()).ToArray();
        Require(components.Length is >= 1 and <= 4 && components.All(float.IsFinite),
            $"{label} must contain 1-4 finite values");
        return components.Length switch
        {
            1 => Broadcast(components[0]),
            2 => new float4(components[0], components[1], 0.0f, 1.0f),
            3 => new float4(components[0], components[1], components[2], 1.0f),
            _ => new float4(components[0], components[1], components[2], components[3])
        };
    }

    private static float4 Broadcast(float value)
    {
        Require(float.IsFinite(value), "material expression value must be finite");
        return new float4(value, value, value, value);
    }

    private static int GradientCode(string value) => value switch
    {
        "LINEAR" => 0,
        "QUADRATIC" => 1,
        "EASING" => 2,
        "DIAGONAL" => 3,
        "SPHERICAL" => 4,
        "QUADRATIC_SPHERE" => 5,
        _ => throw Error($"Gradient Texture mode '{value}' is unsupported by raster evaluation")
    };

    private static int RampInterpolationCode(string value) => value switch
    {
        "CONSTANT" => 0,
        "LINEAR" => 1,
        "EASE" => 2,
        _ => throw Error($"ColorRamp interpolation '{value}' is unsupported by raster evaluation")
    };

    private static int MathCode(string value) => value switch
    {
        "ADD" => 0, "SUBTRACT" => 1, "MULTIPLY" => 2, "DIVIDE" => 3,
        "MULTIPLY_ADD" => 4, "POWER" => 5, "MINIMUM" => 6, "MAXIMUM" => 7,
        "LESS_THAN" => 8, "GREATER_THAN" => 9, "ABSOLUTE" => 10, "SQRT" => 11,
        "FLOOR" => 12, "CEIL" => 13, "FRACT" => 14, "MODULO" => 15,
        "SINE" => 16, "COSINE" => 17, "TANGENT" => 18, "SIGN" => 19,
        "COMPARE" => 20, "PINGPONG" => 21, "SNAP" => 22, "WRAP" => 23,
        _ => throw Error($"Math operation '{value}' is unsupported by raster evaluation")
    };

    private static int VectorMathCode(string value) => value switch
    {
        "ADD" => 0, "SUBTRACT" => 1, "MULTIPLY" => 2, "DIVIDE" => 3,
        "CROSS_PRODUCT" => 4, "DOT_PRODUCT" => 5, "DISTANCE" => 6, "LENGTH" => 7,
        "SCALE" => 8, "NORMALIZE" => 9, "ABSOLUTE" => 10, "MINIMUM" => 11,
        "MAXIMUM" => 12, "FLOOR" => 13, "CEIL" => 14, "FRACTION" => 15,
        "MODULO" => 16, "SINE" => 17, "COSINE" => 18, "TANGENT" => 19,
        _ => throw Error($"Vector Math operation '{value}' is unsupported by raster evaluation")
    };

    private static int MixRgbCode(string value) => value switch
    {
        "MIX" => 0, "ADD" => 1, "MULTIPLY" => 2, "SUBTRACT" => 3,
        "SCREEN" => 4, "DIVIDE" => 5, "DIFFERENCE" => 6, "DARKEN" => 7,
        "LIGHTEN" => 8, "OVERLAY" => 9,
        _ => throw Error($"Mix Color mode '{value}' is unsupported by raster evaluation")
    };

    private static int MappingCode(string value) => value switch
    {
        "POINT" => 0, "TEXTURE" => 1, "VECTOR" => 2, "NORMAL" => 3,
        _ => throw Error($"Mapping vector type '{value}' is unsupported by raster evaluation")
    };

    private static JsonElement Required(JsonElement owner, string name)
    {
        Require(owner.TryGetProperty(name, out var value), $"materialExpression is missing '{name}'");
        return value;
    }

    private static string ReadString(JsonElement owner, string name)
    {
        var value = Required(owner, name);
        Require(value.ValueKind == JsonValueKind.String, $"materialExpression '{name}' must be a string");
        return value.GetString() ?? "";
    }

    private static string ReadOptionalString(JsonElement owner, string name, string fallback)
        => owner.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? fallback
            : fallback;

    private static int ReadInt(JsonElement owner, string name)
    {
        var value = Required(owner, name);
        var result = 0;
        Require(value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out result),
            $"materialExpression '{name}' must be an integer");
        return result;
    }

    private static float ReadFloat(JsonElement owner, string name)
    {
        var value = Required(owner, name);
        Require(value.ValueKind == JsonValueKind.Number, $"materialExpression '{name}' must be numeric");
        var result = value.GetSingle();
        Require(float.IsFinite(result), $"materialExpression '{name}' must be finite");
        return result;
    }

    private static bool ReadOptionalBool(JsonElement owner, string name, bool fallback)
        => owner.TryGetProperty(name, out var value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? value.GetBoolean()
            : fallback;

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw Error(message);
        }
    }

    private static InvalidDataException Error(string message)
        => new($"MATERIAL_EXPRESSION_UNSUPPORTED: {message}");
}
