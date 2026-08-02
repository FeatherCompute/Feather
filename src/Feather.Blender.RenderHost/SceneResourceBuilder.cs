using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost;

internal sealed record RenderSceneResources(
    RenderGeometry Geometry,
    SceneMaterialTable Materials,
    SceneTextureTable Textures,
    SceneLightTable Lights,
    RenderTime Time);

internal static class SceneResourceBuilder
{
    private const string DefaultMaterialId = "__feather_default_material";

    public static RenderSceneResources Build(SceneSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);

        var (textures, textureIndices) = BuildTextures(snapshot);
        var (materials, materialIndices) = BuildMaterials(snapshot, textureIndices, textures);
        var geometry = SceneGeometryBuilder.BuildResolved(
            snapshot,
            materialIndices,
            materials.DefaultMaterialIndex);
        return new RenderSceneResources(
            geometry,
            materials,
            textures,
            BuildLights(snapshot.Metadata.Lights),
            BuildTime(snapshot.Metadata.Frame, snapshot.Metadata.Subframe));
    }

    private static (SceneTextureTable Table, Dictionary<string, int> Indices) BuildTextures(
        SceneSnapshot snapshot)
    {
        var metadata = snapshot.Metadata.Textures
            ?? throw new InvalidDataException("Scene metadata textures are missing.");
        var textures = new SceneTexture[metadata.Length];
        var indices = new Dictionary<string, int>(metadata.Length, StringComparer.Ordinal);
        for (var index = 0; index < metadata.Length; index++)
        {
            var item = metadata[index]
                ?? throw new InvalidDataException("Scene metadata contains a null texture.");
            if (string.IsNullOrWhiteSpace(item.TextureId) || !indices.TryAdd(item.TextureId, index))
            {
                throw new InvalidDataException(
                    $"Scene contains a missing or duplicate texture ID '{item.TextureId}'.");
            }
            if (item.Width <= 0 || item.Height <= 0)
            {
                throw new InvalidDataException($"Scene texture '{item.TextureId}' has invalid dimensions.");
            }
            if (item.Channels is not (0 or 4))
            {
                throw new InvalidDataException($"Scene texture '{item.TextureId}' must contain four channels.");
            }
            if (!string.IsNullOrEmpty(item.ComponentType) &&
                !string.Equals(item.ComponentType, "uint8", StringComparison.Ordinal))
            {
                throw new InvalidDataException($"Scene texture '{item.TextureId}' must use uint8 components.");
            }
            if (snapshot.Metadata.SchemaVersion >= 2 &&
                (item.Channels != 4 ||
                 !string.Equals(item.ComponentType, "uint8", StringComparison.Ordinal) ||
                 !string.Equals(item.Format, "rgba8-unorm", StringComparison.Ordinal) ||
                 item.ContentHash is null || item.ContentHash.Length != 64 ||
                 item.ContentHash.Any(character => character is not (>= '0' and <= '9') and
                                                   not (>= 'a' and <= 'f'))))
            {
                throw new InvalidDataException(
                    $"Scene texture '{item.TextureId}' does not satisfy the v2 RGBA8 payload contract.");
            }

            var bytes = snapshot.ReadUInt8(
                item.Pixels,
                $"{item.TextureId}.pixels",
                item.Height,
                item.Width,
                4);
            var pixels = new Rgba8[checked(item.Width * item.Height)];
            for (var pixelIndex = 0; pixelIndex < pixels.Length; pixelIndex++)
            {
                var byteIndex = pixelIndex * 4;
                pixels[pixelIndex] = new Rgba8(
                    bytes[byteIndex],
                    bytes[byteIndex + 1],
                    bytes[byteIndex + 2],
                    bytes[byteIndex + 3]);
            }
            textures[index] = new SceneTexture(
                item.TextureId,
                item.Name ?? "",
                item.Width,
                item.Height,
                pixels,
                item.ColorSpace ?? "",
                item.AlphaMode ?? "",
                item.Source ?? "",
                item.ContentHash ?? "",
                item.Format ?? "",
                item.Origin ?? "",
                item.IsData,
                item.Packed);
        }
        return (new SceneTextureTable(textures), indices);
    }

    private static (SceneMaterialTable Table, Dictionary<string, int> Indices) BuildMaterials(
        SceneSnapshot snapshot,
        IReadOnlyDictionary<string, int> textureIndices,
        SceneTextureTable textures)
    {
        var metadata = snapshot.Metadata.Materials
            ?? throw new InvalidDataException("Scene metadata materials are missing.");
        var materials = new List<SceneMaterial>(metadata.Length + 1);
        var indices = new Dictionary<string, int>(metadata.Length, StringComparer.Ordinal);
        foreach (var item in metadata)
        {
            if (item is null)
            {
                throw new InvalidDataException("Scene metadata contains a null material.");
            }
            if (string.IsNullOrWhiteSpace(item.MaterialId) ||
                !indices.TryAdd(item.MaterialId, materials.Count))
            {
                throw new InvalidDataException(
                    $"Scene contains a missing or duplicate material ID '{item.MaterialId}'.");
            }

            var baseColor = ReadColor(
                item.BaseColor,
                item.DiffuseColor,
                $"Material '{item.MaterialId}' baseColor");
            var emissionColor = item.EmissionColor is null or { Length: 0 }
                ? new float4(0.0f, 0.0f, 0.0f, 1.0f)
                : ReadColor(item.EmissionColor, null, $"Material '{item.MaterialId}' emissionColor");
            var status = SceneMaterialStatus.Supported;
            var diagnostic = item.Diagnostic;
            if (string.IsNullOrWhiteSpace(item.GraphStatus))
            {
                if (snapshot.Metadata.SchemaVersion >= 2)
                {
                    status = SceneMaterialStatus.Fallback;
                    diagnostic = MergeDiagnostic(diagnostic, "Material graph status is missing.");
                }
            }
            else
            {
                if (string.Equals(item.GraphStatus, "fallback", StringComparison.OrdinalIgnoreCase))
                {
                    status = SceneMaterialStatus.Fallback;
                }
                else if (!string.Equals(item.GraphStatus, "supported", StringComparison.OrdinalIgnoreCase))
                {
                    status = SceneMaterialStatus.Fallback;
                    diagnostic = MergeDiagnostic(
                        diagnostic,
                        $"Unknown material graph status '{item.GraphStatus}'.");
                }
            }

            var textureIndex = SceneMaterial.NoTexture;
            if (!string.IsNullOrWhiteSpace(item.BaseColorTextureId))
            {
                if (!textureIndices.TryGetValue(item.BaseColorTextureId, out textureIndex))
                {
                    status = SceneMaterialStatus.Fallback;
                    diagnostic = MergeDiagnostic(
                        diagnostic,
                        $"Base color texture '{item.BaseColorTextureId}' is missing.");
                    textureIndex = SceneMaterial.NoTexture;
                }
            }
            if (textureIndex >= textures.Textures.Length)
            {
                throw new InvalidDataException(
                    $"Material '{item.MaterialId}' resolved an invalid texture table index.");
            }

            SceneMaterialExpression? expression = null;
            if (status == SceneMaterialStatus.Supported)
            {
                try
                {
                    expression = MaterialExpressionCompiler.Compile(
                        item.MaterialExpression,
                        textureIndices);
                    if (expression is not null && expression.TextureIndex >= textures.Textures.Length)
                    {
                        throw new InvalidDataException(
                            "MATERIAL_EXPRESSION_UNSUPPORTED: resolved texture index is invalid");
                    }
                }
                catch (InvalidDataException exception)
                {
                    status = SceneMaterialStatus.Fallback;
                    diagnostic = MergeDiagnostic(diagnostic, exception.Message);
                }
            }

            var metallic = item.Metallic ?? 0.0f;
            var roughness = item.Roughness ?? 0.5f;
            var ior = item.Ior ?? SceneMaterial.DefaultIor;
            var diffuseRoughness = item.DiffuseRoughness ?? SceneMaterial.DefaultDiffuseRoughness;
            var transmissionWeight = item.TransmissionWeight ?? SceneMaterial.DefaultTransmissionWeight;
            var sheenWeight = item.SheenWeight ?? SceneMaterial.DefaultSheenWeight;
            var sheenColor = item.SheenColor is null or { Length: 0 }
                ? SceneMaterial.DefaultSheenColor
                : ReadColor(item.SheenColor, null, $"Material '{item.MaterialId}' sheenColor");
            var clearcoatWeight = item.ClearcoatWeight ?? SceneMaterial.DefaultClearcoatWeight;
            var clearcoatRoughness = item.ClearcoatRoughness ?? SceneMaterial.DefaultClearcoatRoughness;
            var alpha = item.Alpha ?? baseColor.W;
            var emissionStrength = item.EmissionStrength ?? 0.0f;
            ValidatePrincipledRange(item.MaterialId, "ior", ior, 1.0f, 1000.0f);
            ValidatePrincipledRange(
                item.MaterialId,
                "diffuseRoughness",
                diffuseRoughness,
                0.0f,
                1.0f);
            ValidatePrincipledRange(
                item.MaterialId,
                "transmissionWeight",
                transmissionWeight,
                0.0f,
                1.0f);
            ValidatePrincipledRange(item.MaterialId, "sheenWeight", sheenWeight, 0.0f, 1.0f);
            ValidatePrincipledRange(item.MaterialId, "clearcoatWeight", clearcoatWeight, 0.0f, 1.0f);
            ValidatePrincipledRange(item.MaterialId, "clearcoatRoughness", clearcoatRoughness, 0.0f, 1.0f);
            if (status == SceneMaterialStatus.Fallback)
            {
                diagnostic = string.IsNullOrWhiteSpace(diagnostic)
                    ? "The Blender material graph uses unsupported nodes or links."
                    : diagnostic;
                baseColor = SceneMaterial.FallbackBaseColor;
                metallic = 0.0f;
                roughness = 1.0f;
                ior = SceneMaterial.DefaultIor;
                diffuseRoughness = SceneMaterial.DefaultDiffuseRoughness;
                transmissionWeight = SceneMaterial.DefaultTransmissionWeight;
                sheenWeight = SceneMaterial.DefaultSheenWeight;
                sheenColor = SceneMaterial.DefaultSheenColor;
                clearcoatWeight = SceneMaterial.DefaultClearcoatWeight;
                clearcoatRoughness = SceneMaterial.DefaultClearcoatRoughness;
                emissionColor = new float4(0.0f, 0.0f, 0.0f, 1.0f);
                emissionStrength = 0.0f;
                alpha = 1.0f;
                textureIndex = SceneMaterial.NoTexture;
                expression = null;
            }

            try
            {
                materials.Add(new SceneMaterial(
                    item.MaterialId,
                    item.Name ?? item.MaterialId,
                    baseColor,
                    metallic,
                    roughness,
                    emissionColor,
                    alpha,
                    ior,
                    diffuseRoughness,
                    transmissionWeight,
                    textureIndex,
                    status,
                    diagnostic,
                    emissionStrength,
                    sheenWeight,
                    sheenColor,
                    clearcoatWeight,
                    clearcoatRoughness,
                    expression));
            }
            catch (ArgumentException exception)
            {
                throw new InvalidDataException(
                    $"Scene material '{item.MaterialId}' is invalid: {exception.Message}",
                    exception);
            }
        }

        var defaultMaterialIndex = materials.Count;
        materials.Add(new SceneMaterial(
            DefaultMaterialId,
            "Default Material",
            new float4(0.8f, 0.8f, 0.8f, 1.0f),
            0.0f,
            0.5f,
            new float4(0.0f, 0.0f, 0.0f, 1.0f),
            1.0f));
        return (new SceneMaterialTable(materials.ToArray(), defaultMaterialIndex, textures), indices);
    }

    private static void ValidatePrincipledRange(
        string materialId,
        string property,
        float value,
        float minimum,
        float maximum)
    {
        if (!float.IsFinite(value) || value < minimum || value > maximum)
        {
            throw new InvalidDataException(
                $"Scene material '{materialId}' has invalid {property}: expected a finite value " +
                $"between {minimum} and {maximum}.");
        }
    }

    private static SceneLightTable BuildLights(SceneLightMetadata[]? metadata)
    {
        if (metadata is null)
        {
            throw new InvalidDataException("Scene metadata lights are missing.");
        }
        var lights = new SceneLight[metadata.Length];
        for (var index = 0; index < metadata.Length; index++)
        {
            var item = metadata[index]
                ?? throw new InvalidDataException("Scene metadata contains a null light.");
            if (item.MatrixWorld is null || item.MatrixWorld.Length != 16)
            {
                throw new InvalidDataException($"Scene light '{item.Name}' has an invalid matrixWorld.");
            }
            if (item.Color is null || item.Color.Length != 3)
            {
                throw new InvalidDataException($"Scene light '{item.Name}' has an invalid color.");
            }
            var type = item.Type switch
            {
                "POINT" => SceneLightType.Point,
                "SUN" => SceneLightType.Directional,
                "SPOT" => SceneLightType.Spot,
                "AREA" => SceneLightType.Area,
                _ => SceneLightType.Unknown
            };
            try
            {
                var position = ReadOptionalFloat3(item.Position, $"Scene light '{item.Name}' position");
                var direction = ReadOptionalFloat3(item.Direction, $"Scene light '{item.Name}' direction");
                lights[index] = new SceneLight(
                    item.Name ?? "",
                    type,
                    MatrixProtocol.FromRowMajor(item.MatrixWorld),
                    new float3(item.Color[0], item.Color[1], item.Color[2]),
                    item.Energy,
                    item.Radius,
                    item.SpotSize,
                    item.SpotBlend,
                    item.LightId ?? "",
                    position,
                    direction,
                    item.AreaShape,
                    item.AreaSize,
                    item.AreaSizeY);
            }
            catch (ArgumentException exception)
            {
                throw new InvalidDataException(
                    $"Scene light '{item.Name}' is invalid: {exception.Message}",
                    exception);
            }
        }
        return new SceneLightTable(lights);
    }

    private static float4 ReadColor(
        IReadOnlyList<float>? primary,
        IReadOnlyList<float>? legacy,
        string name)
    {
        var values = primary is { Count: 4 } ? primary : legacy;
        if (values is null || values.Count != 4 || values.Any(value => !float.IsFinite(value)))
        {
            throw new InvalidDataException($"{name} must contain four finite values.");
        }
        return new float4(values[0], values[1], values[2], values[3]);
    }

    private static float3? ReadOptionalFloat3(IReadOnlyList<float>? values, string name)
    {
        if (values is null or { Count: 0 })
        {
            return null;
        }
        if (values.Count != 3 || values.Any(value => !float.IsFinite(value)))
        {
            throw new InvalidDataException($"{name} must contain three finite values.");
        }
        return new float3(values[0], values[1], values[2]);
    }

    private static RenderTime BuildTime(int frame, float subframe)
    {
        try
        {
            return new RenderTime(frame, subframe);
        }
        catch (ArgumentOutOfRangeException exception)
        {
            throw new InvalidDataException($"Scene timeline position is invalid: {exception.Message}", exception);
        }
    }

    private static string MergeDiagnostic(string? current, string addition)
        => string.IsNullOrWhiteSpace(current) ? addition : $"{current} {addition}";
}
