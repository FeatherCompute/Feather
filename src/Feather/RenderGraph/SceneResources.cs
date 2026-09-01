using Feather.Math;

namespace Feather.RenderGraph;

/// <summary>
/// A contiguous indexed draw range and the scene material used by that range.
/// </summary>
public readonly record struct SceneSubmesh(int FirstIndex, int IndexCount, int MaterialIndex);

/// <summary>
/// Result of translating the Blender material graph subset understood by the host.
/// </summary>
public enum SceneMaterialStatus
{
    Supported = 0,
    Fallback = 1
}

/// <summary>Operations understood by the raster material-expression evaluator.</summary>
public enum SceneMaterialExpressionOp
{
    Constant = 0,
    Uv = 1,
    ImageColor = 2,
    ImageAlpha = 3,
    Noise = 4,
    Voronoi = 5,
    Gradient = 6,
    Checker = 7,
    Mix = 8,
    ColorRamp = 9,
    RgbCurves = 10,
    Math = 11,
    VectorMath = 12,
    MapRange = 13,
    MixRgb = 14,
    HueSaturationValue = 15,
    Mapping = 16,
    NormalMap = 17,
    SeparateXyz = 18,
    CombineXyz = 19,
    Fresnel = 20,
    LayerWeight = 21,
    MixShader = 22,
    AddShader = 23,
    Bump = 24,
    BumpEvaluated = 25
}

/// <summary>
/// Bounded straight-line material topologies understood by the shared raster/path shader evaluator.
/// Unsupported graphs deliberately remain on <see cref="Fallback"/> so shader source size is
/// independent of the number of materials in a scene.
/// </summary>
public enum SceneMaterialCompiledVariant
{
    Fallback = 0,
    Constant = 1,
    TextureChannels = 2,
    TextureMultiply = 3,
    TextureAdd = 4,
    TextureSubtractFromConstant = 5,
    TextureMultiplyAdd = 6,
    TextureMix = 7,
    TextureRampConstant = 8,
    TextureRampLinear = 9,
    TextureRampEase = 10
}

/// <summary>
/// Parameters for one canonical material topology. Texture slots contain resolved scene texture
/// indices; values stay in buffers so all material instances share the same small set of functions.
/// </summary>
public sealed class CompiledMaterialProgram
{
    /// <summary>
    /// The fixed evaluator ABI currently contains ten shader topologies. This is deliberately below
    /// the 16-variant pipeline budget from the GI optimization plan; material count cannot increase it.
    /// </summary>
    public const int VariantCount = 10;

    public CompiledMaterialProgram(
        SceneMaterialCompiledVariant variant,
        int texture0 = SceneMaterial.NoTexture,
        int texture1 = SceneMaterial.NoTexture,
        int texture2 = SceneMaterial.NoTexture,
        int texture3 = SceneMaterial.NoTexture,
        int texture4 = SceneMaterial.NoTexture,
        int channel0 = 0,
        int channel1 = 0,
        int channel2 = 0,
        int channel3 = 0,
        int channel4 = 0,
        int target = 0,
        float4 parameter0 = default,
        float4 parameter1 = default,
        float4 parameter2 = default,
        float4 parameter3 = default)
    {
        if (variant is <= SceneMaterialCompiledVariant.Fallback ||
            (int)variant > VariantCount)
        {
            throw new ArgumentOutOfRangeException(nameof(variant));
        }
        Variant = variant;
        Texture0 = ValidateTexture(texture0, nameof(texture0));
        Texture1 = ValidateTexture(texture1, nameof(texture1));
        Texture2 = ValidateTexture(texture2, nameof(texture2));
        Texture3 = ValidateTexture(texture3, nameof(texture3));
        Texture4 = ValidateTexture(texture4, nameof(texture4));
        Channel0 = channel0;
        Channel1 = channel1;
        Channel2 = channel2;
        Channel3 = channel3;
        Channel4 = channel4;
        Target = target;
        Parameter0 = parameter0;
        Parameter1 = parameter1;
        Parameter2 = parameter2;
        Parameter3 = parameter3;
    }

    public SceneMaterialCompiledVariant Variant { get; }
    public int Texture0 { get; }
    public int Texture1 { get; }
    public int Texture2 { get; }
    public int Texture3 { get; }
    public int Texture4 { get; }
    public int Channel0 { get; }
    public int Channel1 { get; }
    public int Channel2 { get; }
    public int Channel3 { get; }
    public int Channel4 { get; }
    public int Target { get; }
    public float4 Parameter0 { get; }
    public float4 Parameter1 { get; }
    public float4 Parameter2 { get; }
    public float4 Parameter3 { get; }

    private static int ValidateTexture(int texture, string name)
    {
        if (texture < SceneMaterial.NoTexture)
        {
            throw new ArgumentOutOfRangeException(name);
        }
        return texture;
    }
}

/// <summary>A host-lowered instruction copied into a generated pass's shader-local layout.</summary>
public partial struct SceneMaterialExpressionInstruction
{
    public float4 Value;
    public float4 Parameters;
    public int Op;
    public int A;
    public int B;
    public int C;
    public int D;
    public int E;
    public int F;
    public int G;
    public int H;
    public int ParameterOffset;
    public int ParameterCount;
    public int Reserved;
}

/// <summary>Register indices for the material channels carried by <see cref="SceneMaterial"/>.</summary>
public partial struct SceneMaterialExpressionOutputs
{
    public SceneMaterialExpressionOutputs(
        int baseColor,
        int metallic,
        int roughness,
        int ior,
        int diffuseRoughness,
        int transmissionWeight,
        int sheenWeight,
        int sheenColor,
        int clearcoatWeight,
        int clearcoatRoughness,
        int emissionColor,
        int emissionStrength,
        int alpha,
        int normal)
    {
        BaseColor = baseColor;
        Metallic = metallic;
        Roughness = roughness;
        Ior = ior;
        DiffuseRoughness = diffuseRoughness;
        TransmissionWeight = transmissionWeight;
        SheenWeight = sheenWeight;
        SheenColor = sheenColor;
        ClearcoatWeight = clearcoatWeight;
        ClearcoatRoughness = clearcoatRoughness;
        EmissionColor = emissionColor;
        EmissionStrength = emissionStrength;
        Alpha = alpha;
        Normal = normal;
    }

    public int BaseColor;
    public int Metallic;
    public int Roughness;
    public int Ior;
    public int DiffuseRoughness;
    public int TransmissionWeight;
    public int SheenWeight;
    public int SheenColor;
    public int ClearcoatWeight;
    public int ClearcoatRoughness;
    public int EmissionColor;
    public int EmissionStrength;
    public int Alpha;
    public int Normal;
}

/// <summary>
/// A bounded material-expression program lowered from the scene snapshot IR by the render host.
/// </summary>
public sealed class SceneMaterialExpression
{
    /// <summary>
    /// Maximum register/instruction count. 128 covers layered production-style Principled graphs
    /// while keeping the generated shader register file statically bounded.
    /// </summary>
    public const int MaxInstructions = 128;

    public SceneMaterialExpression(
        string hash,
        ReadOnlyMemory<SceneMaterialExpressionInstruction> instructions,
        ReadOnlyMemory<float4> parameters,
        SceneMaterialExpressionOutputs outputs,
        int textureIndex = SceneMaterial.NoTexture,
        CompiledMaterialProgram? compiledProgram = null)
        : this(
            hash,
            instructions,
            parameters,
            outputs,
            textureIndex == SceneMaterial.NoTexture ? ReadOnlyMemory<int>.Empty : new[] { textureIndex },
            compiledProgram)
    {
    }

    public SceneMaterialExpression(
        string hash,
        ReadOnlyMemory<SceneMaterialExpressionInstruction> instructions,
        ReadOnlyMemory<float4> parameters,
        SceneMaterialExpressionOutputs outputs,
        ReadOnlyMemory<int> textureIndices,
        CompiledMaterialProgram? compiledProgram = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(hash);
        if (instructions.IsEmpty || instructions.Length > MaxInstructions)
        {
            throw new ArgumentOutOfRangeException(
                nameof(instructions),
                $"Material expressions must contain between 1 and {MaxInstructions} instructions.");
        }
        foreach (var textureIndex in textureIndices.Span)
        {
            if (textureIndex < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(textureIndices));
            }
        }

        Hash = hash;
        Instructions = instructions;
        Parameters = parameters;
        Outputs = outputs;
        TextureIndices = textureIndices;
        CompiledProgram = compiledProgram;
    }

    public string Hash { get; }

    public ReadOnlyMemory<SceneMaterialExpressionInstruction> Instructions { get; }

    public ReadOnlyMemory<float4> Parameters { get; }

    public SceneMaterialExpressionOutputs Outputs { get; }

    /// <summary>Scene texture bindings in the material-local table order declared by the IR.</summary>
    public ReadOnlyMemory<int> TextureIndices { get; }

    /// <summary>The canonical straight-line lowering, or null when the bounded VM is required.</summary>
    public CompiledMaterialProgram? CompiledProgram { get; }

    /// <summary>Legacy alias for the first material texture binding.</summary>
    public int TextureIndex => TextureIndices.IsEmpty ? SceneMaterial.NoTexture : TextureIndices.Span[0];

    public bool HasTexture => !TextureIndices.IsEmpty;
}

/// <summary>
/// Renderer-independent material values extracted from an evaluated scene.
/// </summary>
public sealed class SceneMaterial
{
    public const int NoTexture = -1;
    public const float DefaultIor = 1.5f;
    public const float DefaultDiffuseRoughness = 0.0f;
    public const float DefaultTransmissionWeight = 0.0f;
    public const float DefaultSheenWeight = 0.0f;
    public static float4 DefaultSheenColor => new(1.0f, 1.0f, 1.0f, 1.0f);
    public const float DefaultClearcoatWeight = 0.0f;
    public const float DefaultClearcoatRoughness = 0.03f;

    public static float4 FallbackBaseColor => new(1.0f, 0.0f, 1.0f, 1.0f);

    public SceneMaterial(
        string id,
        string name,
        float4 baseColor,
        float metallic,
        float roughness,
        float4 emissionColor,
        float alpha,
        int baseColorTextureIndex = NoTexture,
        SceneMaterialStatus status = SceneMaterialStatus.Supported,
        string? diagnostic = null,
        float emissionStrength = 0.0f)
        : this(
            id,
            name,
            baseColor,
            metallic,
            roughness,
            emissionColor,
            alpha,
            DefaultIor,
            DefaultDiffuseRoughness,
            DefaultTransmissionWeight,
            baseColorTextureIndex,
            status,
            diagnostic,
            emissionStrength,
            DefaultSheenWeight,
            DefaultSheenColor,
            DefaultClearcoatWeight,
            DefaultClearcoatRoughness)
    {
    }

    public SceneMaterial(
        string id,
        string name,
        float4 baseColor,
        float metallic,
        float roughness,
        float4 emissionColor,
        float alpha,
        float ior,
        float diffuseRoughness,
        float transmissionWeight,
        int baseColorTextureIndex = NoTexture,
        SceneMaterialStatus status = SceneMaterialStatus.Supported,
        string? diagnostic = null,
        float emissionStrength = 0.0f,
        float sheenWeight = DefaultSheenWeight,
        float4? sheenColor = null,
        float clearcoatWeight = DefaultClearcoatWeight,
        float clearcoatRoughness = DefaultClearcoatRoughness,
        SceneMaterialExpression? expression = null,
        float specular = 0.5f,
        int normalTextureIndex = NoTexture,
        int metallicTextureIndex = NoTexture,
        int roughnessTextureIndex = NoTexture,
        int opacityTextureIndex = NoTexture)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentNullException.ThrowIfNull(name);
        ValidateFinite(baseColor, nameof(baseColor));
        ValidateFinite(emissionColor, nameof(emissionColor));
        if (!float.IsFinite(metallic) || metallic is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(metallic), "Metallic must be between zero and one.");
        }
        if (!float.IsFinite(roughness) || roughness is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(roughness), "Roughness must be between zero and one.");
        }
        if (!float.IsFinite(specular) || specular is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(specular), "Specular must be between zero and one.");
        }
        if (!float.IsFinite(ior) || ior is < 1.0f or > 1000.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(ior), "IOR must be between one and 1000.");
        }
        if (!float.IsFinite(diffuseRoughness) || diffuseRoughness is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(diffuseRoughness),
                "Diffuse roughness must be between zero and one.");
        }
        if (!float.IsFinite(transmissionWeight) || transmissionWeight is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(transmissionWeight),
                "Transmission weight must be between zero and one.");
        }
        if (!float.IsFinite(sheenWeight) || sheenWeight is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(sheenWeight), "Sheen weight must be between zero and one.");
        }
        var resolvedSheenColor = sheenColor ?? DefaultSheenColor;
        ValidateFinite(resolvedSheenColor, nameof(sheenColor));
        if (!float.IsFinite(clearcoatWeight) || clearcoatWeight is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(clearcoatWeight), "Clearcoat weight must be between zero and one.");
        }
        if (!float.IsFinite(clearcoatRoughness) || clearcoatRoughness is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(clearcoatRoughness), "Clearcoat roughness must be between zero and one.");
        }
        if (!float.IsFinite(alpha) || alpha is < 0.0f or > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be between zero and one.");
        }
        if (!float.IsFinite(emissionStrength))
        {
            throw new ArgumentOutOfRangeException(nameof(emissionStrength));
        }
        if (baseColorTextureIndex < NoTexture || normalTextureIndex < NoTexture ||
            metallicTextureIndex < NoTexture || roughnessTextureIndex < NoTexture ||
            opacityTextureIndex < NoTexture)
        {
            throw new ArgumentOutOfRangeException(
                nameof(baseColorTextureIndex),
                "Material texture indices must be -1 or greater.");
        }
        if (!Enum.IsDefined(status))
        {
            throw new ArgumentOutOfRangeException(nameof(status));
        }
        if (status == SceneMaterialStatus.Fallback && string.IsNullOrWhiteSpace(diagnostic))
        {
            throw new ArgumentException("A fallback material must include a diagnostic.", nameof(diagnostic));
        }

        Id = id;
        Name = name;
        BaseColor = baseColor;
        Metallic = metallic;
        Specular = specular;
        Roughness = roughness;
        Ior = ior;
        DiffuseRoughness = diffuseRoughness;
        TransmissionWeight = transmissionWeight;
        SheenWeight = sheenWeight;
        SheenColor = resolvedSheenColor;
        ClearcoatWeight = clearcoatWeight;
        ClearcoatRoughness = clearcoatRoughness;
        EmissionColor = emissionColor;
        EmissionStrength = emissionStrength;
        Alpha = alpha;
        BaseColorTextureIndex = baseColorTextureIndex;
        NormalTextureIndex = normalTextureIndex;
        MetallicTextureIndex = metallicTextureIndex;
        RoughnessTextureIndex = roughnessTextureIndex;
        OpacityTextureIndex = opacityTextureIndex;
        Status = status;
        Diagnostic = diagnostic;
        Expression = expression;
    }

    public string Id { get; }

    public string Name { get; }

    public float4 BaseColor { get; }

    public float Metallic { get; }

    public float Specular { get; }

    public float Roughness { get; }

    public float Ior { get; }

    public float DiffuseRoughness { get; }

    public float TransmissionWeight { get; }

    public float SheenWeight { get; }

    public float4 SheenColor { get; }

    public float ClearcoatWeight { get; }

    public float ClearcoatRoughness { get; }

    public float4 EmissionColor { get; }

    /// <summary>
    /// Original Principled emission strength retained for inspection. Blender snapshot exporters
    /// fold this value into <see cref="EmissionColor"/>; renderers must not multiply it again.
    /// </summary>
    public float EmissionStrength { get; }

    public float Alpha { get; }

    public int BaseColorTextureIndex { get; }

    public int NormalTextureIndex { get; }

    public int MetallicTextureIndex { get; }

    public int RoughnessTextureIndex { get; }

    public int OpacityTextureIndex { get; }

    public SceneMaterialStatus Status { get; }

    public string? Diagnostic { get; }

    /// <summary>The per-pixel expression program, or null for the classic flattened material path.</summary>
    public SceneMaterialExpression? Expression { get; }

    public bool HasBaseColorTexture => BaseColorTextureIndex != NoTexture;

    public bool HasExpression => Expression is not null;

    private static void ValidateFinite(float4 value, string name)
    {
        if (!float.IsFinite(value.X) || !float.IsFinite(value.Y) ||
            !float.IsFinite(value.Z) || !float.IsFinite(value.W))
        {
            throw new ArgumentOutOfRangeException(name, "Material colors must contain finite values.");
        }
    }
}

/// <summary>
/// Read-only material table owned by the render host.
/// </summary>
public sealed class SceneMaterialTable
{
    public SceneMaterialTable(
        ReadOnlyMemory<SceneMaterial> materials,
        int defaultMaterialIndex,
        SceneTextureTable? textures = null)
    {
        if (materials.IsEmpty)
        {
            throw new ArgumentException("A scene material table must contain a default material.", nameof(materials));
        }
        if ((uint)defaultMaterialIndex >= (uint)materials.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(defaultMaterialIndex));
        }
        foreach (var material in materials.Span)
        {
            if (material is null)
            {
                throw new ArgumentException("Scene materials cannot contain null entries.", nameof(materials));
            }
        }

        Materials = materials;
        DefaultMaterialIndex = defaultMaterialIndex;
        Textures = textures ?? new SceneTextureTable(Array.Empty<SceneTexture>());
    }

    public ReadOnlyMemory<SceneMaterial> Materials { get; }

    public int DefaultMaterialIndex { get; }

    /// <summary>Textures referenced by the compiled expressions in this material table.</summary>
    /// <remarks>
    /// Keeping the references here lets a renderer that shades several materials in one dispatch
    /// build one shared atlas. Raster passes can still consume the standalone texture-table resource.
    /// </remarks>
    public SceneTextureTable Textures { get; }
}

/// <summary>
/// Tightly packed RGBA8 image data extracted from a Blender image data-block.
/// </summary>
public sealed class SceneTexture
{
    public SceneTexture(
        string id,
        string name,
        int width,
        int height,
        ReadOnlyMemory<Rgba8> pixels,
        string colorSpace,
        string alphaMode,
        string source,
        string contentHash,
        string format = "rgba8-unorm",
        string origin = "bottom-left",
        bool isData = false,
        bool packed = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(colorSpace);
        ArgumentNullException.ThrowIfNull(alphaMode);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(contentHash);
        ArgumentNullException.ThrowIfNull(format);
        ArgumentNullException.ThrowIfNull(origin);
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }
        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }
        if (pixels.Length != checked(width * height))
        {
            throw new ArgumentException("Texture pixels must contain exactly width * height RGBA8 values.", nameof(pixels));
        }

        Id = id;
        Name = name;
        Width = width;
        Height = height;
        Pixels = pixels;
        ColorSpace = colorSpace;
        AlphaMode = alphaMode;
        Source = source;
        ContentHash = contentHash;
        Format = format;
        Origin = origin;
        IsData = isData;
        Packed = packed;
    }

    public string Id { get; }

    public string Name { get; }

    public int Width { get; }

    public int Height { get; }

    public ReadOnlyMemory<Rgba8> Pixels { get; }

    public string ColorSpace { get; }

    public string AlphaMode { get; }

    public string Source { get; }

    public string ContentHash { get; }

    public string Format { get; }

    public string Origin { get; }

    public bool IsData { get; }

    public bool Packed { get; }
}

/// <summary>
/// Read-only image table owned by the render host.
/// </summary>
public sealed class SceneTextureTable
{
    public SceneTextureTable(ReadOnlyMemory<SceneTexture> textures)
    {
        foreach (var texture in textures.Span)
        {
            if (texture is null)
            {
                throw new ArgumentException("Scene textures cannot contain null entries.", nameof(textures));
            }
        }
        Textures = textures;
    }

    public ReadOnlyMemory<SceneTexture> Textures { get; }
}

public enum SceneLightType
{
    Unknown = 0,
    Point = 1,
    Directional = 2,
    Spot = 3,
    Area = 4
}

/// <summary>
/// Evaluated Blender light data in world space.
/// </summary>
public sealed class SceneLight
{
    public SceneLight(
        string name,
        SceneLightType type,
        float4x4 transform,
        float3 color,
        float energy,
        float radius,
        float spotSize,
        float spotBlend,
        string id = "",
        float3? position = null,
        float3? direction = null,
        string? areaShape = null,
        float areaSize = 0.0f,
        float areaSizeY = 0.0f)
    {
        ArgumentNullException.ThrowIfNull(name);
        if (!Enum.IsDefined(type))
        {
            throw new ArgumentOutOfRangeException(nameof(type));
        }
        if (!IsFinite(transform) || !IsFinite(color) ||
            !float.IsFinite(energy) || !float.IsFinite(radius) ||
            !float.IsFinite(spotSize) || !float.IsFinite(spotBlend) ||
            !float.IsFinite(areaSize) || !float.IsFinite(areaSizeY) ||
            (position.HasValue && !IsFinite(position.Value)) ||
            (direction.HasValue && !IsFinite(direction.Value)))
        {
            throw new ArgumentOutOfRangeException(nameof(transform), "Light values must be finite.");
        }

        Name = name;
        Id = string.IsNullOrWhiteSpace(id) ? name : id;
        Type = type;
        Transform = transform;
        Color = color;
        Energy = energy;
        Radius = radius;
        SpotSize = spotSize;
        SpotBlend = spotBlend;
        Position = position ?? new float3(transform.M03, transform.M13, transform.M23);
        Direction = NormalizeDirection(
            direction ?? new float3(-transform.M02, -transform.M12, -transform.M22));
        AreaShape = areaShape;
        AreaSize = areaSize;
        AreaSizeY = areaSizeY;
    }

    public string Id { get; }

    public string Name { get; }

    public SceneLightType Type { get; }

    public float4x4 Transform { get; }

    public float3 Color { get; }

    public float Energy { get; }

    public float Radius { get; }

    public float SpotSize { get; }

    public float SpotBlend { get; }

    public float3 Position { get; }

    /// <summary>
    /// Blender lights point down their local negative Z axis.
    /// </summary>
    public float3 Direction { get; }

    public string? AreaShape { get; }

    public float AreaSize { get; }

    public float AreaSizeY { get; }

    private static float3 NormalizeDirection(float3 value)
    {
        var length = MathF.Sqrt((value.X * value.X) + (value.Y * value.Y) + (value.Z * value.Z));
        return length > 1e-8f ? value / length : new float3(0.0f, 0.0f, -1.0f);
    }

    private static bool IsFinite(float3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);

    private static bool IsFinite(float4x4 value)
        => IsFinite(value.C0) && IsFinite(value.C1) && IsFinite(value.C2) && IsFinite(value.C3);

    private static bool IsFinite(float4 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) &&
           float.IsFinite(value.Z) && float.IsFinite(value.W);
}

/// <summary>
/// Read-only evaluated light table owned by the render host.
/// </summary>
public sealed class SceneLightTable
{
    public SceneLightTable(ReadOnlyMemory<SceneLight> lights)
    {
        foreach (var light in lights.Span)
        {
            if (light is null)
            {
                throw new ArgumentException("Scene lights cannot contain null entries.", nameof(lights));
            }
        }
        Lights = lights;
    }

    public ReadOnlyMemory<SceneLight> Lights { get; }
}

/// <summary>The execution domain retained for one Actor in an evaluated Scene.</summary>
public enum RenderSceneActorDomain
{
    Triangle = 0,
    Analytic = 1,
    SignedDistance = 2,
}

/// <summary>
/// One immutable Actor passed through a <see cref="RenderScene"/> without forcing it into a mesh.
/// <c>RepresentationId</c> is a renderer-visible geometry discriminator; it is not an Asset Type
/// name and does not require an adapter registry. A renderer may switch on it directly, while a
/// graph-authored adapter pass remains an optional user implementation.
/// </summary>
public sealed class RenderSceneActor
{
    public RenderSceneActor(
        string id,
        string representationId,
        RenderSceneActorDomain domain,
        float3 center,
        float3 size,
        float3 rotationDegrees,
        float radius,
        float3 normal,
        int materialIndex,
        ReadOnlyMemory<float> parameters = default,
        SceneGeometry? triangleGeometry = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentException.ThrowIfNullOrWhiteSpace(representationId);
        if (!Enum.IsDefined(domain)) throw new ArgumentOutOfRangeException(nameof(domain));
        if (!IsFinite(center) || !IsFinite(size) || !IsFinite(rotationDegrees) || !IsFinite(normal) ||
            !float.IsFinite(radius) || radius < 0 || materialIndex < 0)
        {
            throw new ArgumentException("Scene Actor values must be finite and bounded.");
        }
        foreach (float parameter in parameters.Span)
        {
            if (!float.IsFinite(parameter))
                throw new ArgumentException("Scene Actor representation parameters must be finite.", nameof(parameters));
        }
        if (domain == RenderSceneActorDomain.Triangle && triangleGeometry is null)
            throw new ArgumentException("Triangle-domain Actors require geometry.", nameof(triangleGeometry));
        if (domain != RenderSceneActorDomain.Triangle && triangleGeometry is not null)
            throw new ArgumentException("Only triangle-domain Actors may carry triangle geometry.", nameof(triangleGeometry));

        Id = id;
        RepresentationId = representationId;
        Domain = domain;
        Center = center;
        Size = size;
        RotationDegrees = rotationDegrees;
        Radius = radius;
        Normal = normal;
        MaterialIndex = materialIndex;
        Parameters = parameters;
        TriangleGeometry = triangleGeometry;
    }

    public string Id { get; }
    public string RepresentationId { get; }
    public RenderSceneActorDomain Domain { get; }
    public float3 Center { get; }
    public float3 Size { get; }
    public float3 RotationDegrees { get; }
    public float Radius { get; }
    public float3 Normal { get; }
    public int MaterialIndex { get; }
    public ReadOnlyMemory<float> Parameters { get; }
    public SceneGeometry? TriangleGeometry { get; }

    private static bool IsFinite(float3 value) =>
        float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}

/// <summary>
/// Immutable evaluated Scene object supplied as one graph resource. It intentionally retains
/// heterogeneous Actor representations so a renderer can implement native hybrid execution.
/// </summary>
public sealed class RenderScene
{
    public RenderScene(
        string id,
        ReadOnlyMemory<RenderSceneActor> actors,
        SceneMaterialTable materials,
        SceneTextureTable textures,
        SceneLightTable lights,
        RenderCamera camera)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentNullException.ThrowIfNull(materials);
        ArgumentNullException.ThrowIfNull(textures);
        ArgumentNullException.ThrowIfNull(lights);
        foreach (RenderSceneActor actor in actors.Span)
        {
            if (actor is null) throw new ArgumentException("Scene Actors cannot contain null entries.", nameof(actors));
            if (actor.MaterialIndex >= materials.Materials.Length)
                throw new ArgumentException("Scene Actor material index is out of range.", nameof(actors));
        }
        Id = id;
        Actors = actors;
        Materials = materials;
        Textures = textures;
        Lights = lights;
        Camera = camera;
    }

    public string Id { get; }
    public ReadOnlyMemory<RenderSceneActor> Actors { get; }
    public SceneMaterialTable Materials { get; }
    public SceneTextureTable Textures { get; }
    public SceneLightTable Lights { get; }
    public RenderCamera Camera { get; }
}

/// <summary>
/// Evaluated Blender timeline position for a render request.
/// </summary>
public readonly record struct RenderTime
{
    public RenderTime(int frame, float subframe)
    {
        if (!float.IsFinite(subframe) || subframe is < 0.0f or >= 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(subframe), "Subframe must be in the range [0, 1).");
        }
        Frame = frame;
        Subframe = subframe;
    }

    public int Frame { get; }

    public float Subframe { get; }
}
