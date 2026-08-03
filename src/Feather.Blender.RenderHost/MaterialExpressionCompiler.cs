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

        var expressionTextureIds = new List<string>();
        var expressionTextureIndices = new List<int>();
        if (root.TryGetProperty("textures", out var textureTable))
        {
            Require(textureTable.ValueKind == JsonValueKind.Array,
                "materialExpression textures must be an array");
            foreach (var texture in textureTable.EnumerateArray())
            {
                Require(texture.ValueKind == JsonValueKind.String,
                    "materialExpression texture table entries must be strings");
                var id = texture.GetString() ?? "";
                Require(!string.IsNullOrWhiteSpace(id),
                    "materialExpression texture table entries cannot be empty");
                Require(!expressionTextureIds.Contains(id, StringComparer.Ordinal),
                    $"materialExpression texture table contains duplicate texture '{id}'");
                Require(textureIndices.TryGetValue(id, out var resolvedTexture),
                    $"materialExpression references missing texture '{id}'");
                expressionTextureIds.Add(id);
                expressionTextureIndices.Add(resolvedTexture);
            }
        }

        var indices = new Dictionary<string, int>(nodes.GetArrayLength(), StringComparer.Ordinal);
        var instructions = new List<SceneMaterialExpressionInstruction>(nodes.GetArrayLength());
        var parameters = new List<float4>();
        foreach (var node in nodes.EnumerateArray())
        {
            Require(node.ValueKind == JsonValueKind.Object, "materialExpression node must be an object");
            var id = ReadString(node, "id");
            Require(!indices.ContainsKey(id), $"materialExpression contains duplicate node ID '{id}'");
            var instruction = CompileNode(
                node,
                indices,
                parameters,
                textureIndices,
                expressionTextureIds,
                expressionTextureIndices);
            Require(indices.TryAdd(id, instructions.Count),
                $"materialExpression contains duplicate node ID '{id}'");
            instructions.Add(instruction);
        }
        var outputs = Required(root, "outputs");
        Require(outputs.ValueKind == JsonValueKind.Object, "materialExpression outputs must be an object");
        int Output(string name) => ResolveReference(ReadString(outputs, name), indices, name);
        var expressionOutputs = new SceneMaterialExpressionOutputs(
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
            Output("normal"));
        LowerBump(instructions, ref expressionOutputs);
        var compiledProgram = CompileCanonical(instructions, parameters, expressionOutputs);
        return new SceneMaterialExpression(
            hash,
            instructions.ToArray(),
            parameters.ToArray(),
            expressionOutputs,
            expressionTextureIndices.ToArray(),
            compiledProgram);
    }

    /// <summary>
    /// Recognizes a deliberately small topology vocabulary. This is a second lowering target rather
    /// than source generation per material: a scene with 287 instances still emits the same ten
    /// shader functions, while texture IDs and constants remain data.
    /// </summary>
    private static CompiledMaterialProgram? CompileCanonical(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        IReadOnlyList<float4> parameters,
        SceneMaterialExpressionOutputs outputs)
    {
        if (AllOutputsConstant(instructions, outputs))
        {
            // Keep scalar-only Principled graphs on the compatibility path. Besides being cheap, the
            // established gates intentionally pin their extreme-IOR floating-point behaviour. The
            // independently removable cost is in texture/math programs, not these leaf constants.
            return null;
        }

        if (TryCompileTextureChannels(instructions, outputs, out var channels))
        {
            return channels;
        }

        if (TryCompileTextureMath(instructions, outputs, out var math))
        {
            return math;
        }

        if (TryCompileTextureMix(instructions, outputs, out var mix))
        {
            return mix;
        }

        if (TryCompileTextureRamp(instructions, parameters, outputs, out var ramp))
        {
            return ramp;
        }

        return null;
    }

    private static bool TryCompileTextureChannels(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        SceneMaterialExpressionOutputs outputs,
        out CompiledMaterialProgram? program)
    {
        program = null;
        if (!OtherOutputsAreConstant(instructions, outputs) ||
            !TryOptionalColor(instructions, outputs.BaseColor, out var baseColor) ||
            !TryOptionalScalar(instructions, outputs.Metallic, out var metallic) ||
            !TryOptionalScalar(instructions, outputs.Roughness, out var roughness) ||
            !TryOptionalScalar(instructions, outputs.Alpha, out var alpha) ||
            !TryOptionalNormal(instructions, outputs.Normal, out var normal))
        {
            return false;
        }

        program = new CompiledMaterialProgram(
            SceneMaterialCompiledVariant.TextureChannels,
            baseColor.Texture,
            metallic.Texture,
            roughness.Texture,
            alpha.Texture,
            normal.Texture,
            baseColor.Channel,
            metallic.Channel,
            roughness.Channel,
            alpha.Channel,
            normal.Channel,
            parameter0: new float4(normal.Strength, 0.0f, 0.0f, 0.0f));
        return true;
    }

    private static bool TryCompileTextureMath(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        SceneMaterialExpressionOutputs outputs,
        out CompiledMaterialProgram? program)
    {
        program = null;
        if (!OtherOutputsAreConstant(instructions, outputs) ||
            !TryOptionalColor(instructions, outputs.BaseColor, out var baseColor) ||
            !TryOptionalNormal(instructions, outputs.Normal, out var normal))
        {
            return false;
        }

        var dynamicCount = 0;
        var target = 0;
        var output = -1;
        CountDynamic(outputs.Metallic, 1);
        CountDynamic(outputs.Roughness, 2);
        CountDynamic(outputs.Alpha, 3);
        if (dynamicCount != 1 || !TryMathChain(instructions, output, out var math))
        {
            return false;
        }

        // Non-target scalar channels must be constants. Direct image channels are handled by the
        // texture-channel topology and a graph with two unrelated math chains stays on the VM.
        if ((target == 1 || IsConstant(instructions, outputs.Metallic)) &&
            (target == 2 || IsConstant(instructions, outputs.Roughness)) &&
            (target == 3 || IsConstant(instructions, outputs.Alpha)))
        {
            program = new CompiledMaterialProgram(
                math.Variant,
                baseColor.Texture,
                math.Source.Texture,
                texture4: normal.Texture,
                channel0: baseColor.Channel,
                channel1: math.Source.Channel,
                channel4: normal.Channel,
                target: target,
                parameter0: math.Parameters,
                parameter1: new float4(normal.Strength, 0.0f, 0.0f, 0.0f));
            return true;
        }
        return false;

        void CountDynamic(int index, int candidateTarget)
        {
            if (!IsConstant(instructions, index))
            {
                dynamicCount++;
                target = candidateTarget;
                output = index;
            }
        }
    }

    private static bool TryCompileTextureMix(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        SceneMaterialExpressionOutputs outputs,
        out CompiledMaterialProgram? program)
    {
        program = null;
        if (!AllExceptBaseColorConstant(instructions, outputs))
        {
            return false;
        }
        var instruction = instructions[outputs.BaseColor];
        if (instruction.Op is not ((int)SceneMaterialExpressionOp.Mix) and
            not ((int)SceneMaterialExpressionOp.MixRgb))
        {
            return false;
        }
        var blendMode = instruction.Op == (int)SceneMaterialExpressionOp.Mix
            ? instruction.Parameters.Z
            : instruction.Parameters.X;
        var clampResult = instruction.Op == (int)SceneMaterialExpressionOp.Mix
            ? instruction.Parameters.Y
            : instruction.Parameters.Y;
        if (blendMode != 0.0f || clampResult > 0.5f ||
            !TryColorOrConstant(instructions, instruction.B, out var a) ||
            !TryColorOrConstant(instructions, instruction.C, out var b) ||
            !TryScalarOrConstant(instructions, instruction.A, out var factor))
        {
            return false;
        }
        var clampFactor = instruction.Op == (int)SceneMaterialExpressionOp.Mix
            ? instruction.Parameters.X
            : 1.0f;
        program = new CompiledMaterialProgram(
            SceneMaterialCompiledVariant.TextureMix,
            a.Source.Texture,
            b.Source.Texture,
            factor.Source.Texture,
            channel0: a.Source.Channel,
            channel1: b.Source.Channel,
            channel2: factor.Source.Channel,
            parameter0: a.Constant,
            parameter1: b.Constant,
            parameter2: new float4(factor.Constant.X, clampFactor, 0.0f, 0.0f));
        return true;
    }

    private static bool TryCompileTextureRamp(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        IReadOnlyList<float4> parameters,
        SceneMaterialExpressionOutputs outputs,
        out CompiledMaterialProgram? program)
    {
        program = null;
        if (!AllExceptBaseColorConstant(instructions, outputs))
        {
            return false;
        }
        var instruction = instructions[outputs.BaseColor];
        if (instruction.Op != (int)SceneMaterialExpressionOp.ColorRamp ||
            instruction.ParameterCount != 2 || instruction.Parameters.Y > 0.5f ||
            !TryScalarOrConstant(instructions, instruction.A, out var factor))
        {
            return false;
        }
        var offset = instruction.ParameterOffset;
        if (offset < 0 || offset + 3 >= parameters.Count)
        {
            return false;
        }
        var variant = instruction.Parameters.X switch
        {
            0.0f => SceneMaterialCompiledVariant.TextureRampConstant,
            1.0f => SceneMaterialCompiledVariant.TextureRampLinear,
            2.0f => SceneMaterialCompiledVariant.TextureRampEase,
            _ => SceneMaterialCompiledVariant.Fallback
        };
        if (variant == SceneMaterialCompiledVariant.Fallback)
        {
            return false;
        }
        program = new CompiledMaterialProgram(
            variant,
            factor.Source.Texture,
            channel0: factor.Source.Channel,
            parameter0: factor.Constant,
            parameter1: parameters[offset],
            parameter2: parameters[offset + 2],
            parameter3: new float4(parameters[offset + 1].X, parameters[offset + 3].X, 0.0f, 0.0f));
        return true;
    }

    private static bool TryMathChain(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int output,
        out MathChain chain)
    {
        chain = default;
        if ((uint)output >= (uint)instructions.Count)
        {
            return false;
        }
        var outer = instructions[output];
        if (outer.Op != (int)SceneMaterialExpressionOp.Math)
        {
            return false;
        }

        // A multiply followed by an add is common for mask remapping. It is one fixed fused topology,
        // not a two-record interpreter loop.
        if ((int)outer.Parameters.X == 0 && outer.Parameters.Y <= 0.5f)
        {
            var innerIndex = IsConstant(instructions, outer.A) ? outer.B : outer.A;
            var addIndex = innerIndex == outer.A ? outer.B : outer.A;
            if (TryConstant(instructions, addIndex, out var add) &&
                (uint)innerIndex < (uint)instructions.Count)
            {
                var inner = instructions[innerIndex];
                if (inner.Op == (int)SceneMaterialExpressionOp.Math &&
                    (int)inner.Parameters.X == 2 && inner.Parameters.Y <= 0.5f &&
                    TryBinaryTextureConstant(instructions, inner, out var source, out var multiply))
                {
                    chain = new MathChain(
                        SceneMaterialCompiledVariant.TextureMultiplyAdd,
                        source,
                        new float4(multiply.X, add.X, 0.0f, 0.0f));
                    return true;
                }
            }
        }

        var operation = (int)outer.Parameters.X;
        if (operation is 0 or 2 &&
            TryBinaryTextureConstant(instructions, outer, out var commutativeSource, out var constant))
        {
            chain = new MathChain(
                operation == 2
                    ? SceneMaterialCompiledVariant.TextureMultiply
                    : SceneMaterialCompiledVariant.TextureAdd,
                commutativeSource,
                new float4(constant.X, outer.Parameters.Y, 0.0f, 0.0f));
            return true;
        }
        if (operation == 1 && TryConstant(instructions, outer.A, out var from) &&
            TryScalarSource(instructions, outer.B, out var subtractSource))
        {
            chain = new MathChain(
                SceneMaterialCompiledVariant.TextureSubtractFromConstant,
                subtractSource,
                new float4(from.X, outer.Parameters.Y, 0.0f, 0.0f));
            return true;
        }
        return false;
    }

    private static bool TryBinaryTextureConstant(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        SceneMaterialExpressionInstruction instruction,
        out TextureSource source,
        out float4 constant)
    {
        if (TryScalarSource(instructions, instruction.A, out source) &&
            TryConstant(instructions, instruction.B, out constant))
        {
            return true;
        }
        if (TryScalarSource(instructions, instruction.B, out source) &&
            TryConstant(instructions, instruction.A, out constant))
        {
            return true;
        }
        source = TextureSource.None;
        constant = float4.Zero;
        return false;
    }

    private static bool TryOptionalColor(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index,
        out TextureSource source)
    {
        if (IsConstant(instructions, index))
        {
            source = TextureSource.None;
            return true;
        }
        return TryColorSource(instructions, index, out source);
    }

    private static bool TryOptionalScalar(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index,
        out TextureSource source)
    {
        if (IsConstant(instructions, index))
        {
            source = TextureSource.None;
            return true;
        }
        return TryScalarSource(instructions, index, out source);
    }

    private static bool TryOptionalNormal(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index,
        out NormalSource source)
    {
        if (TryConstant(instructions, index, out var constant))
        {
            // The flattened SceneMaterial has no tangent-normal field. Only the zero sentinel can be
            // left to its defaults; an authored constant normal must retain exact VM evaluation.
            source = NormalSource.None;
            return constant.X == 0.0f && constant.Y == 0.0f && constant.Z == 0.0f;
        }
        if ((uint)index >= (uint)instructions.Count)
        {
            source = NormalSource.None;
            return false;
        }
        var normal = instructions[index];
        if (normal.Op == (int)SceneMaterialExpressionOp.NormalMap &&
            TryColorSource(instructions, normal.A, out var image) &&
            TryConstant(instructions, normal.B, out var strength))
        {
            source = new NormalSource(image.Texture, image.Channel, ShaderMathClampNonNegative(strength.X));
            return true;
        }
        source = NormalSource.None;
        return false;
    }

    private static bool TryColorSource(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index,
        out TextureSource source)
    {
        if ((uint)index < (uint)instructions.Count)
        {
            var item = instructions[index];
            if (item.Op == (int)SceneMaterialExpressionOp.ImageColor && IsUv(instructions, item.A))
            {
                source = new TextureSource(item.Reserved, -1);
                return true;
            }
        }
        source = TextureSource.None;
        return false;
    }

    private static bool TryScalarSource(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index,
        out TextureSource source)
    {
        if ((uint)index < (uint)instructions.Count)
        {
            var item = instructions[index];
            if (item.Op == (int)SceneMaterialExpressionOp.ImageAlpha && IsUv(instructions, item.A))
            {
                source = new TextureSource(item.Reserved, 3);
                return true;
            }
            if (item.Op == (int)SceneMaterialExpressionOp.ImageColor && IsUv(instructions, item.A))
            {
                source = new TextureSource(item.Reserved, 0);
                return true;
            }
            if (item.Op == (int)SceneMaterialExpressionOp.SeparateXyz &&
                TryColorSource(instructions, item.A, out var image))
            {
                source = new TextureSource(image.Texture, (int)item.Parameters.X);
                return true;
            }
        }
        source = TextureSource.None;
        return false;
    }

    private static bool TryColorOrConstant(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index,
        out ValueSource value)
    {
        if (TryConstant(instructions, index, out var constant))
        {
            value = new ValueSource(TextureSource.None, constant);
            return true;
        }
        if (TryColorSource(instructions, index, out var source))
        {
            value = new ValueSource(source, float4.Zero);
            return true;
        }
        value = default;
        return false;
    }

    private static bool TryScalarOrConstant(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index,
        out ValueSource value)
    {
        if (TryConstant(instructions, index, out var constant))
        {
            value = new ValueSource(TextureSource.None, constant);
            return true;
        }
        if (TryScalarSource(instructions, index, out var source))
        {
            value = new ValueSource(source, float4.Zero);
            return true;
        }
        value = default;
        return false;
    }

    private static bool TryConstant(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index,
        out float4 value)
    {
        if (IsConstant(instructions, index))
        {
            value = instructions[index].Value;
            return true;
        }
        value = float4.Zero;
        return false;
    }

    private static bool IsConstant(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index)
        => (uint)index < (uint)instructions.Count &&
           instructions[index].Op == (int)SceneMaterialExpressionOp.Constant;

    private static bool IsUv(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        int index)
        => (uint)index < (uint)instructions.Count &&
           instructions[index].Op == (int)SceneMaterialExpressionOp.Uv;

    private static bool AllOutputsConstant(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        SceneMaterialExpressionOutputs outputs)
        => OutputIndices(outputs).All(index => IsConstant(instructions, index));

    private static bool OtherOutputsAreConstant(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        SceneMaterialExpressionOutputs outputs)
        => IsConstant(instructions, outputs.Ior) &&
           IsConstant(instructions, outputs.DiffuseRoughness) &&
           IsConstant(instructions, outputs.TransmissionWeight) &&
           IsConstant(instructions, outputs.SheenWeight) &&
           IsConstant(instructions, outputs.SheenColor) &&
           IsConstant(instructions, outputs.ClearcoatWeight) &&
           IsConstant(instructions, outputs.ClearcoatRoughness) &&
           IsConstant(instructions, outputs.EmissionColor) &&
           IsConstant(instructions, outputs.EmissionStrength);

    private static bool AllExceptBaseColorConstant(
        IReadOnlyList<SceneMaterialExpressionInstruction> instructions,
        SceneMaterialExpressionOutputs outputs)
        => IsConstant(instructions, outputs.Metallic) &&
           IsConstant(instructions, outputs.Roughness) &&
           IsConstant(instructions, outputs.Ior) &&
           IsConstant(instructions, outputs.DiffuseRoughness) &&
           IsConstant(instructions, outputs.TransmissionWeight) &&
           IsConstant(instructions, outputs.SheenWeight) &&
           IsConstant(instructions, outputs.SheenColor) &&
           IsConstant(instructions, outputs.ClearcoatWeight) &&
           IsConstant(instructions, outputs.ClearcoatRoughness) &&
           IsConstant(instructions, outputs.EmissionColor) &&
           IsConstant(instructions, outputs.EmissionStrength) &&
           IsConstant(instructions, outputs.Alpha) &&
           IsConstant(instructions, outputs.Normal);

    private static int[] OutputIndices(SceneMaterialExpressionOutputs outputs) =>
    [
        outputs.BaseColor,
        outputs.Metallic,
        outputs.Roughness,
        outputs.Ior,
        outputs.DiffuseRoughness,
        outputs.TransmissionWeight,
        outputs.SheenWeight,
        outputs.SheenColor,
        outputs.ClearcoatWeight,
        outputs.ClearcoatRoughness,
        outputs.EmissionColor,
        outputs.EmissionStrength,
        outputs.Alpha,
        outputs.Normal
    ];

    private static float ShaderMathClampNonNegative(float value) => System.MathF.Max(value, 0.0f);

    private readonly record struct TextureSource(int Texture, int Channel)
    {
        public static TextureSource None => new(SceneMaterial.NoTexture, 0);
    }

    private readonly record struct NormalSource(int Texture, int Channel, float Strength)
    {
        public static NormalSource None => new(SceneMaterial.NoTexture, 0, 1.0f);
    }

    private readonly record struct ValueSource(TextureSource Source, float4 Constant);

    private readonly record struct MathChain(
        SceneMaterialCompiledVariant Variant,
        TextureSource Source,
        float4 Parameters);

    private static void LowerBump(
        List<SceneMaterialExpressionInstruction> instructions,
        ref SceneMaterialExpressionOutputs outputs)
    {
        var original = instructions.ToArray();
        var remapped = new int[original.Length];
        instructions.Clear();

        for (var index = 0; index < original.Length; index++)
        {
            var instruction = RemapInputs(original[index], remapped);
            if (instruction.Op == (int)SceneMaterialExpressionOp.Bump)
            {
                var height = original[original[index].A];
                if (height.Op is (int)SceneMaterialExpressionOp.ImageColor
                    or (int)SceneMaterialExpressionOp.ImageAlpha)
                {
                    var coordinate = remapped[height.A];
                    var delta = 1.0f / 256.0f;
                    var left = AddBumpTap(instructions, height, coordinate, -delta, 0.0f);
                    var right = AddBumpTap(instructions, height, coordinate, delta, 0.0f);
                    var down = AddBumpTap(instructions, height, coordinate, 0.0f, -delta);
                    var up = AddBumpTap(instructions, height, coordinate, 0.0f, delta);
                    var derivatives = AddCombine(instructions, left, right, down);
                    var controls = AddCombine(instructions, up, instruction.B, instruction.C);
                    instruction.E = derivatives;
                    instruction.F = controls;
                    instruction.Parameters = new float4(
                        instruction.Parameters.X, 1.0f, 0.0f, 0.0f);
                }
            }
            remapped[index] = instructions.Count;
            instructions.Add(instruction);
        }

        Require(instructions.Count <= SceneMaterialExpression.MaxInstructions,
            $"lowered materialExpression exceeds {SceneMaterialExpression.MaxInstructions} instructions");
        outputs.BaseColor = remapped[outputs.BaseColor];
        outputs.Metallic = remapped[outputs.Metallic];
        outputs.Roughness = remapped[outputs.Roughness];
        outputs.Ior = remapped[outputs.Ior];
        outputs.DiffuseRoughness = remapped[outputs.DiffuseRoughness];
        outputs.TransmissionWeight = remapped[outputs.TransmissionWeight];
        outputs.SheenWeight = remapped[outputs.SheenWeight];
        outputs.SheenColor = remapped[outputs.SheenColor];
        outputs.ClearcoatWeight = remapped[outputs.ClearcoatWeight];
        outputs.ClearcoatRoughness = remapped[outputs.ClearcoatRoughness];
        outputs.EmissionColor = remapped[outputs.EmissionColor];
        outputs.EmissionStrength = remapped[outputs.EmissionStrength];
        outputs.Alpha = remapped[outputs.Alpha];
        outputs.Normal = remapped[outputs.Normal];
    }

    private static SceneMaterialExpressionInstruction RemapInputs(
        SceneMaterialExpressionInstruction instruction,
        int[] remapped)
    {
        int Remap(int value) => value < 0 ? value : remapped[value];
        instruction.A = Remap(instruction.A);
        instruction.B = Remap(instruction.B);
        instruction.C = Remap(instruction.C);
        instruction.D = Remap(instruction.D);
        instruction.E = Remap(instruction.E);
        instruction.F = Remap(instruction.F);
        instruction.G = Remap(instruction.G);
        instruction.H = Remap(instruction.H);
        return instruction;
    }

    private static int AddBumpTap(
        List<SceneMaterialExpressionInstruction> instructions,
        SceneMaterialExpressionInstruction height,
        int coordinate,
        float offsetX,
        float offsetY)
    {
        var tap = EmptyInstruction();
        tap.Op = height.Op;
        tap.A = coordinate;
        tap.Parameters = new float4(0.0f, 0.0f, offsetX, offsetY);
        tap.Reserved = height.Reserved;
        instructions.Add(tap);
        return instructions.Count - 1;
    }

    private static int AddCombine(
        List<SceneMaterialExpressionInstruction> instructions,
        int x,
        int y,
        int z)
    {
        var combine = EmptyInstruction();
        combine.Op = (int)SceneMaterialExpressionOp.CombineXyz;
        combine.A = x;
        combine.B = y;
        combine.C = z;
        instructions.Add(combine);
        return instructions.Count - 1;
    }

    private static SceneMaterialExpressionInstruction CompileNode(
        JsonElement node,
        IReadOnlyDictionary<string, int> indices,
        List<float4> parameters,
        IReadOnlyDictionary<string, int> textureIndices,
        List<string> expressionTextureIds,
        List<int> expressionTextureIndices)
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
                var localTextureIndex = expressionTextureIds.FindIndex(
                    candidate => string.Equals(candidate, id, StringComparison.Ordinal));
                if (nodeParameters.TryGetProperty("textureIndex", out var textureIndexValue))
                {
                    Require(textureIndexValue.ValueKind == JsonValueKind.Number &&
                            textureIndexValue.TryGetInt32(out localTextureIndex),
                        "image_texture textureIndex must be an integer");
                    Require((uint)localTextureIndex < (uint)expressionTextureIds.Count,
                        $"image_texture textureIndex {localTextureIndex} is outside the material texture table");
                    Require(string.Equals(
                            expressionTextureIds[localTextureIndex], id, StringComparison.Ordinal),
                        $"image_texture textureIndex {localTextureIndex} does not reference texture '{id}'");
                    resolvedTexture = expressionTextureIndices[localTextureIndex];
                }
                else if (localTextureIndex < 0)
                {
                    // Legacy version-1 expressions had only textureId on the node and no root table.
                    localTextureIndex = expressionTextureIds.Count;
                    expressionTextureIds.Add(id);
                    expressionTextureIndices.Add(resolvedTexture);
                }
                // The IR index is material-local; the lowered instruction carries the scene texture
                // binding used by both atlas-backed evaluators.
                instruction.Reserved = resolvedTexture;
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
                    MixRgbCode(ReadOptionalString(nodeParameters, "blend_type", "MIX")),
                    0.0f);
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
                var mathOperation = ReadString(nodeParameters, "operation");
                instruction.Op = (int)SceneMaterialExpressionOp.Math;
                instruction.A = Input("Value");
                instruction.B = IsUnaryMath(mathOperation) ? instruction.A : Input("Value_001");
                instruction.C = inputs.TryGetProperty("Value_002", out _) ? Input("Value_002") : instruction.A;
                instruction.Parameters = new float4(
                    MathCode(mathOperation),
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
                instruction.Op = (int)SceneMaterialExpressionOp.Bump;
                instruction.A = Input("Height");
                instruction.B = Input("Strength");
                instruction.C = Input("Distance");
                instruction.D = Input("Normal");
                instruction.Parameters = new float4(
                    ReadOptionalBool(nodeParameters, "invert", false) ? 1.0f : 0.0f,
                    0.0f,
                    0.0f,
                    0.0f);
                break;
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
        "RADIAL" => 6,
        _ => throw Error($"Gradient Texture mode '{value}' is unsupported by raster evaluation")
    };

    private static bool IsUnaryMath(string value) => value is
        "ABSOLUTE" or "SQRT" or "FLOOR" or "CEIL" or "FRACT" or "SINE" or
        "COSINE" or "TANGENT" or "SIGN" or "ARCTANGENT";

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
        "ARCTANGENT" => 24,
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
