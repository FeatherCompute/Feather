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

/// <summary>
/// Renderer-independent material values extracted from an evaluated scene.
/// </summary>
public sealed class SceneMaterial
{
    public const int NoTexture = -1;

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
        EmissionColor = emissionColor;
        EmissionStrength = emissionStrength;
        Alpha = alpha;
        BaseColorTextureIndex = baseColorTextureIndex;
        Status = status;
        Diagnostic = diagnostic;
    }

    public string Id { get; }

    public string Name { get; }

    public float4 BaseColor { get; }

    public float Metallic { get; }

    public float Roughness { get; }

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

    public bool HasBaseColorTexture => BaseColorTextureIndex != NoTexture;

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
    public SceneMaterialTable(ReadOnlyMemory<SceneMaterial> materials, int defaultMaterialIndex)
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
    }

    public ReadOnlyMemory<SceneMaterial> Materials { get; }

    public int DefaultMaterialIndex { get; }
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
