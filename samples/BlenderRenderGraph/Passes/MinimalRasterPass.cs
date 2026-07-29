using Feather;
using Feather.Graphics;
using Feather.Math;
using Feather.RenderGraph;
using Feather.Resources;

namespace BlenderRenderGraph.Passes;

[FeatherPass(
    "01c671a1-9b4e-5cab-b7e1-c101348af596",
    Name = "Minimal Raster",
    Category = "Raster",
    Version = 2)]
public sealed class MinimalRasterPass : IRasterPass
{
    [Input("6d6eb2d5-bb7a-55a4-a85a-c58e36715c53")]
    public SceneGeometryHandle Geometry { get; init; }

    [Input("a6eed590-b632-5f91-a69d-09b6eb4bb5ac")]
    public MaterialTableHandle Materials { get; init; }

    [Input("f0a3d759-8290-4a99-84e2-cd40a5f0c6c6")]
    public TextureTableHandle Textures { get; init; }

    [Input("cc78191c-ac9a-57b6-bcac-91cce5e298f5")]
    public CameraHandle Camera { get; init; }

    [Input("58c14650-59bb-4445-8477-6cf363a0375d")]
    public LightTableHandle Lights { get; init; }

    [Input("45b04c81-4895-4302-901c-cf4ba27f0eef")]
    public TimeHandle Time { get; init; }

    [Output(
        "bd711ea6-36f9-56cd-863a-cfec58727a46",
        Format = TextureFormat.Rgba8)]
    public TextureHandle Color { get; init; }

    [Parameter(
        "2a1df649-2b96-558f-ae30-9d9bf6858d43",
        Min = 0.0,
        Max = 8.0)]
    public float Exposure { get; set; } = 1.0f;

    [Parameter(
        "85012a85-1525-54d6-a933-a22ed40fdb73",
        Min = 0.0,
        Max = 2.0)]
    public int ViewMode { get; set; }

    public void Execute(RenderContext context)
    {
        var geometry = context.GetSceneGeometry(Geometry);
        var materials = context.GetMaterials(Materials);
        var textures = context.GetTextures(Textures);
        var camera = context.GetCamera(Camera);
        var lights = context.GetLights(Lights);
        _ = context.GetTime(Time);

        using var color = GPU.CreateRenderTexture2D<Rgba8, Rgba8>(
            context.Width,
            context.Height,
            PixelFormat.Rgba8);
        var background = new Rgba8[checked(context.Width * context.Height)];
        Array.Fill(background, new Rgba8(9, 12, 15, 255));
        color.Upload(background);

        var dispatchPath = DispatchPath.None;
        if (!geometry.Indices.IsEmpty)
        {
            dispatchPath = DrawScene(context, geometry, materials, textures, camera, lights, color);
        }
        context.SetColorOutput(Color, color, dispatchPath);
    }

    private DispatchPath DrawScene(
        RenderContext context,
        SceneGeometry geometry,
        SceneMaterialTable materials,
        SceneTextureTable textures,
        RenderCamera camera,
        SceneLightTable lights,
        GpuTexture2D<Rgba8, Rgba8> color)
    {
        var shaderVertices = new MinimalRasterVertex[geometry.Vertices.Length];
        for (var index = 0; index < shaderVertices.Length; index++)
        {
            var vertex = geometry.Vertices.Span[index];
            shaderVertices[index] = new MinimalRasterVertex
            {
                Position = vertex.Position,
                Normal = vertex.Normal,
                UV = vertex.UV
            };
        }

        var draws = geometry.Submeshes.IsEmpty
            ? [new SceneSubmesh(0, geometry.Indices.Length, materials.DefaultMaterialIndex)]
            : geometry.Submeshes.ToArray();
        using var vertices = GPU.CreateBuffer(shaderVertices, BufferAccess.ReadOnly);
        using var depth = GPU.CreateDepthTexture2D(context.Width, context.Height);
        using var sampler = GPU.CreateSampler(SamplerDesc.LinearRepeat);
        using var whiteTexture = CreateWhiteTexture();
        using var pipeline = GPU.CreateGraphicsPipeline<
            MinimalRasterVertexShader,
            MinimalRasterFragmentShader,
            MinimalRasterVaryings>(
            new GraphicsPipelineDesc
            {
                SampleCount = context.SampleCount,
                DepthStencil = DepthStencilState.Default with
                {
                    DepthTest = true,
                    DepthWrite = true,
                    DepthCompare = CompareOp.Less
                },
                Raster = RasterState.Default with { CullMode = CullMode.None },
                DebugName = "Project Minimal Raster"
            });

        var gpuTextures = new GpuTexture2D<Rgba8, float4>?[textures.Textures.Length];
        try
        {
            ResolveLight(lights, out var lightType, out var lightPosition, out var lightDirection,
                out var lightColor, out var lightEnergy);
            IGpuTexture2D[] targets = [color];
            var firstDraw = true;
            foreach (var draw in draws)
            {
                var material = materials.Materials.Span[draw.MaterialIndex];
                var texture = whiteTexture.AsSampled();
                if (ViewMode == 2 && material.HasBaseColorTexture)
                {
                    var textureIndex = material.BaseColorTextureIndex;
                    gpuTextures[textureIndex] ??= CreateSceneTexture(textures.Textures.Span[textureIndex]);
                    texture = gpuTextures[textureIndex]!.AsSampled();
                }

                var baseColor = ViewMode == 0
                    ? new float4(0.8f, 0.8f, 0.8f, 1.0f)
                    : material.BaseColor;
                var metallic = ViewMode == 2 ? material.Metallic : 0.0f;
                var roughness = ViewMode == 2 ? material.Roughness : 1.0f;
                var emission = ViewMode == 2
                    ? material.EmissionColor
                    : new float4(0.0f, 0.0f, 0.0f, 1.0f);

                using var indices = GPU.CreateIndexBuffer(
                    geometry.Indices.Span.Slice(draw.FirstIndex, draw.IndexCount));
                pipeline.DrawIndexed(
                    new MinimalRasterVertexShader(
                        vertices.AsReadOnly(),
                        new Uniform<float4x4>(camera.ViewProjection)),
                    new MinimalRasterFragmentShader(
                        texture,
                        sampler,
                        new Uniform<float4>(baseColor),
                        new Uniform<float>(metallic),
                        new Uniform<float>(roughness),
                        new Uniform<float4>(emission),
                        new Uniform<int>(lightType),
                        new Uniform<float3>(lightPosition),
                        new Uniform<float3>(lightDirection),
                        new Uniform<float3>(lightColor),
                        new Uniform<float>(lightEnergy),
                        new Uniform<float>(Exposure),
                        new Uniform<int>(ViewMode)),
                    targets,
                    depth,
                    indices,
                    new GraphicsDrawDesc
                    {
                        ColorLoadOp = firstDraw
                            ? GraphicsColorLoadOp.Clear
                            : GraphicsColorLoadOp.Load,
                        ClearColor = firstDraw
                            ? new float4(0.035f, 0.047f, 0.059f, 1.0f)
                            : null,
                        DepthLoadOp = firstDraw
                            ? GraphicsDepthLoadOp.Clear
                            : GraphicsDepthLoadOp.Load,
                        ClearDepth = firstDraw ? 1.0f : null
                    });
                firstDraw = false;
            }
            return pipeline.LastDispatchPath;
        }
        finally
        {
            foreach (var texture in gpuTextures)
            {
                texture?.Dispose();
            }
        }
    }

    private static GpuTexture2D<Rgba8, float4> CreateWhiteTexture()
    {
        var texture = GPU.CreateTexture2D<Rgba8, float4>(
            1,
            1,
            PixelFormat.Rgba8,
            TextureAccess.Sampled);
        texture.Upload([new Rgba8(255, 255, 255, 255)]);
        return texture;
    }

    private static GpuTexture2D<Rgba8, float4> CreateSceneTexture(SceneTexture source)
    {
        var texture = GPU.CreateTexture2D<Rgba8, float4>(
            source.Width,
            source.Height,
            PixelFormat.Rgba8,
            TextureAccess.Sampled);
        texture.Upload(source.Pixels.Span);
        return texture;
    }

    private static void ResolveLight(
        SceneLightTable lights,
        out int lightType,
        out float3 position,
        out float3 direction,
        out float3 color,
        out float energy)
    {
        SceneLight? selected = null;
        foreach (var light in lights.Lights.Span)
        {
            if (light.Type == SceneLightType.Directional)
            {
                selected = light;
                break;
            }
            if (selected is null && light.Type == SceneLightType.Point)
            {
                selected = light;
            }
        }

        if (selected is null)
        {
            lightType = 0;
            position = float3.Zero;
            var fallback = new float3(0.35f, -0.45f, 0.82f);
            var fallbackLength = MathF.Sqrt(
                (fallback.X * fallback.X) +
                (fallback.Y * fallback.Y) +
                (fallback.Z * fallback.Z));
            direction = fallback / fallbackLength;
            color = new float3(1.0f, 1.0f, 1.0f);
            energy = 1.0f;
            return;
        }

        lightType = selected.Type == SceneLightType.Point ? 1 : 2;
        position = selected.Position;
        direction = -selected.Direction;
        color = selected.Color;
        energy = selected.Energy;
    }
}

[GpuStruct]
public partial struct MinimalRasterVertex
{
    public float3 Position;
    public float3 Normal;
    public float2 UV;
}

[GpuStruct]
public partial struct MinimalRasterVaryings
{
    [Position]
    public float4 Position;
    public float3 WorldPosition;
    public float3 Normal;
    public float2 UV;
}

[VertexShader]
public readonly partial struct MinimalRasterVertexShader(
    ReadOnlyBuffer<MinimalRasterVertex> vertices,
    Uniform<float4x4> viewProjection) : IVertexShader<MinimalRasterVaryings>
{
    public MinimalRasterVaryings Execute()
    {
        var vertex = vertices[VertexIds.Index];
        return new MinimalRasterVaryings
        {
            Position = ShaderMath.Mul(viewProjection.Value, new float4(vertex.Position, 1.0f)),
            WorldPosition = vertex.Position,
            Normal = vertex.Normal,
            UV = vertex.UV
        };
    }
}

[FragmentShader]
public readonly partial struct MinimalRasterFragmentShader(
    SampledTexture2D<float4> baseColorTexture,
    SamplerState sampler,
    Uniform<float4> baseColor,
    Uniform<float> metallic,
    Uniform<float> roughness,
    Uniform<float4> emission,
    Uniform<int> lightType,
    Uniform<float3> lightPosition,
    Uniform<float3> lightDirection,
    Uniform<float3> lightColor,
    Uniform<float> lightEnergy,
    Uniform<float> exposure,
    Uniform<int> viewMode) : IFragmentShader<MinimalRasterVaryings>
{
    public float4 Execute(MinimalRasterVaryings input)
    {
        var normal = ShaderMath.Normalize(input.Normal);
        if (viewMode.Value == 1)
        {
            return new float4((normal * 0.5f) + new float3(0.5f, 0.5f, 0.5f), 1.0f);
        }

        var sampled = baseColorTexture.Sample(sampler, input.UV);
        var surface = new float3(
            sampled.R * baseColor.Value.R,
            sampled.G * baseColor.Value.G,
            sampled.B * baseColor.Value.B);
        var toLight = lightDirection.Value;
        var illumination = ShaderMath.Min(lightEnergy.Value, 4.0f);
        if (lightType.Value == 1)
        {
            var delta = lightPosition.Value - input.WorldPosition;
            var distanceSquared = ShaderMath.Max(ShaderMath.Dot(delta, delta), 0.01f);
            toLight = ShaderMath.Normalize(delta);
            illumination = ShaderMath.Min(lightEnergy.Value / (12.56637f * distanceSquared), 4.0f);
        }

        var diffuse = ShaderMath.Max(ShaderMath.Dot(normal, toLight), 0.0f);
        var dielectric = 1.0f - metallic.Value;
        var diffuseStrength = dielectric * diffuse;
        var specularStrength =
            (0.04f * dielectric + metallic.Value) * (1.0f - roughness.Value * 0.75f) * diffuse;
        var direct = surface * (diffuseStrength * illumination);
        var ambient = surface * (0.12f + roughness.Value * 0.08f);
        var specular = lightColor.Value * (specularStrength * illumination);
        var emitted = new float3(emission.Value.R, emission.Value.G, emission.Value.B);
        var result = (ambient + direct + specular + emitted) * exposure.Value;
        return new float4(result, 1.0f);
    }
}
