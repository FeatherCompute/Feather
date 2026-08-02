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
    AddShader = 23
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
    public const int MaxInstructions = 64;

    public SceneMaterialExpression(
        string hash,
        ReadOnlyMemory<SceneMaterialExpressionInstruction> instructions,
        ReadOnlyMemory<float4> parameters,
        SceneMaterialExpressionOutputs outputs,
        int textureIndex = SceneMaterial.NoTexture)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(hash);
        if (instructions.IsEmpty || instructions.Length > MaxInstructions)
        {
            throw new ArgumentOutOfRangeException(
                nameof(instructions),
                $"Material expressions must contain between 1 and {MaxInstructions} instructions.");
        }
        if (textureIndex < SceneMaterial.NoTexture)
        {
            throw new ArgumentOutOfRangeException(nameof(textureIndex));
        }

        Hash = hash;
        Instructions = instructions;
        Parameters = parameters;
        Outputs = outputs;
        TextureIndex = textureIndex;
    }

    public string Hash { get; }

    public ReadOnlyMemory<SceneMaterialExpressionInstruction> Instructions { get; }

    public ReadOnlyMemory<float4> Parameters { get; }

    public SceneMaterialExpressionOutputs Outputs { get; }

    public int TextureIndex { get; }

    public bool HasTexture => TextureIndex != SceneMaterial.NoTexture;
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
        SceneMaterialExpression? expression = null)
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
        if (baseColorTextureIndex < NoTexture)
        {
            throw new ArgumentOutOfRangeException(nameof(baseColorTextureIndex));
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
        Status = status;
        Diagnostic = diagnostic;
        Expression = expression;
    }

    public string Id { get; }

    public string Name { get; }

    public float4 BaseColor { get; }

    public float Metallic { get; }

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
