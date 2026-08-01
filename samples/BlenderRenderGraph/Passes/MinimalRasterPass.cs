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

    // The same colour the draw clears to, in the render target's 8-bit encoding. Kept next to that
    // clear so the two cannot drift apart.
    private static readonly Rgba8 BackgroundColor = new(9, 12, 15, 255);

    private static readonly float4 ClearColor = new(0.035f, 0.047f, 0.059f, 1.0f);

    public void Execute(RenderContext context)
    {
        var geometry = context.GetSceneGeometry(Geometry);
        var materials = context.GetMaterials(Materials);
        var textures = context.GetTextures(Textures);
        var camera = context.GetCamera(Camera);
        var lights = context.GetLights(Lights);
        _ = context.GetTime(Time);

#if FEATHER_POOLED_RASTER_TARGETS
        var color = context.GetOrCreateRenderTarget<Rgba8, Rgba8>(Color, PixelFormat.Rgba8);
#else
        using var color = GPU.CreateRenderTexture2D<Rgba8, Rgba8>(
            context.Width,
            context.Height,
            PixelFormat.Rgba8);
#endif

        var dispatchPath = DispatchPath.None;
        if (geometry.Indices.IsEmpty)
        {
            // Nothing will be drawn, so no render pass runs to clear the target. Paint the
            // background by hand here instead. When there is geometry the draw clears to the same
            // colour, so doing this unconditionally allocated and uploaded a full-resolution image
            // per frame only to overwrite it.
            var background = new Rgba8[checked(context.Width * context.Height)];
            Array.Fill(background, BackgroundColor);
            color.Upload(background);
        }
        else
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
        var shaderLights = BuildLights(lights);
        using var vertices = GPU.CreateBuffer(shaderVertices, BufferAccess.ReadOnly);
        using var lightBuffer = GPU.CreateBuffer(shaderLights, BufferAccess.ReadOnly);
#if FEATHER_POOLED_RASTER_TARGETS
        var depth = context.GetOrCreateDepthTarget(Color);
#else
        using var depth = GPU.CreateDepthTexture2D(context.Width, context.Height);
#endif
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
            IGpuTexture2D[] targets = [color];
            // Fresnel and the specular lobe are view-dependent, so the eye has to reach the shader.
            var cameraPosition = camera.WorldPosition;
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
                // Only the material preview shades the real material. The white-model and normal
                // views deliberately pin these so geometry reads clearly, which means an IOR of 1.5
                // (the F0 = 0.04 dielectric) and no diffuse roughness or transmission.
                var ior = ViewMode == 2 ? material.Ior : SceneMaterial.DefaultIor;
                var diffuseRoughness = ViewMode == 2 ? material.DiffuseRoughness : 0.0f;
                var transmissionWeight = ViewMode == 2 ? material.TransmissionWeight : 0.0f;
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
                        new Uniform<float>(ior),
                        new Uniform<float>(diffuseRoughness),
                        new Uniform<float>(transmissionWeight),
                        new Uniform<float4>(emission),
                        new Uniform<float>(Exposure),
                        new Uniform<int>(ViewMode),
                        new Uniform<int>(shaderLights.Length),
                        new Uniform<float3>(cameraPosition),
                        lightBuffer.AsReadOnly()),
                    targets,
                    depth,
                    indices,
                    new GraphicsDrawDesc
                    {
                        ColorLoadOp = firstDraw
                            ? GraphicsColorLoadOp.Clear
                            : GraphicsColorLoadOp.Load,
                        ClearColor = firstDraw ? ClearColor : null,
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

    // Every light in the scene reaches the shader; the fragment stage accumulates them in a loop.
    // An empty light table still uploads one entry so the buffer binding always has valid storage,
    // and that entry carries the historical fallback sun so unlit scenes stay readable.
    private static MinimalRasterLight[] BuildLights(SceneLightTable lights)
    {
        if (lights.Lights.IsEmpty)
        {
            var fallback = new float3(0.35f, -0.45f, 0.82f);
            var fallbackLength = MathF.Sqrt(
                (fallback.X * fallback.X) +
                (fallback.Y * fallback.Y) +
                (fallback.Z * fallback.Z));
            return
            [
                new MinimalRasterLight
                {
                    Position = float3.Zero,
                    Kind = DirectionalLightKind,
                    Direction = fallback / fallbackLength,
                    Energy = 1.0f,
                    Color = new float3(1.0f, 1.0f, 1.0f),
                    ConeOuter = 0.0f,
                    ConeInner = 0.0f
                }
            ];
        }

        var shaderLights = new MinimalRasterLight[lights.Lights.Length];
        for (var index = 0; index < shaderLights.Length; index++)
        {
            var light = lights.Lights.Span[index];
            shaderLights[index] = new MinimalRasterLight
            {
                Position = light.Position,
                Kind = MapLightKind(light.Type),
                // Blender aims a light down its local -Z; the shader wants the surface-to-light
                // direction, so the stored vector is flipped once here instead of per fragment.
                Direction = -light.Direction,
                Energy = light.Energy,
                Color = light.Color,
                ConeOuter = MathF.Cos(MathF.Min(light.SpotSize, MathF.PI) * 0.5f),
                ConeInner = SpotConeInner(light)
            };
        }

        return shaderLights;
    }

    private static int MapLightKind(SceneLightType type) => type switch
    {
        SceneLightType.Point => PointLightKind,
        SceneLightType.Directional => DirectionalLightKind,
        SceneLightType.Spot => SpotLightKind,
        SceneLightType.Area => AreaLightKind,
        _ => DirectionalLightKind
    };

    // Blender's spot blend runs 0 (hard edge) to 1 (fully soft), measured inward from the cone
    // edge. Converting it to a second cosine lets the shader fade between the two with one
    // Smoothstep instead of computing angles on the GPU.
    private static float SpotConeInner(SceneLight light)
    {
        var half = MathF.Min(light.SpotSize, MathF.PI) * 0.5f;
        var blend = System.Math.Clamp(light.SpotBlend, 0.0f, 1.0f);
        return MathF.Cos(half * (1.0f - blend));
    }

    private const int PointLightKind = 1;
    private const int DirectionalLightKind = 2;
    private const int SpotLightKind = 3;
    private const int AreaLightKind = 4;
}

// Laid out as float3/scalar pairs so the scalar occupies the vec3 padding slot and the managed
// size matches the std430 stride exactly. Declared beside the pass rather than in the SDK because
// generated projects compile this source against the published package.
[GpuStruct]
public partial struct MinimalRasterLight
{
    public float3 Position;
    public int Kind;
    public float3 Direction;
    public float Energy;
    public float3 Color;
    public float ConeOuter;
    public float ConeInner;
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
    Uniform<float> ior,
    Uniform<float> diffuseRoughness,
    Uniform<float> transmissionWeight,
    Uniform<float4> emission,
    Uniform<float> exposure,
    Uniform<int> viewMode,
    Uniform<int> lightCount,
    Uniform<float3> cameraPosition,
    // The light buffer is declared last so it lands on a high binding: the native bridge infers the
    // vertex stream from the lowest bound buffer and dedupes cross-stage buffers by source binding.
    ReadOnlyBuffer<MinimalRasterLight> lights) : IFragmentShader<MinimalRasterVaryings>
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
        var dielectric = 1.0f - metallic.Value;

        // Normal-incidence reflectance from the index of refraction, via Schlick's parameterisation
        // of Fresnel: F0 = ((n - 1) / (n + 1))^2. The old code hardcoded 0.04, which is glass at
        // n = 1.5 -- correct for exactly one material and the reason the IOR slider did nothing.
        var iorRatio = (ior.Value - 1.0f) / (ior.Value + 1.0f);
        var dielectricF0 = iorRatio * iorRatio;
        // A metal's reflectance is its base colour; a dielectric's is a colourless few percent.
        var f0 = ShaderMath.Lerp(
            new float3(dielectricF0, dielectricF0, dielectricF0),
            surface,
            metallic.Value);

        var view = ShaderMath.Normalize(cameraPosition.Value - input.WorldPosition);
        // Two-sided shading: a back face lit from behind should not go black, and transmission needs
        // the geometric side to stay meaningful.
        if (ShaderMath.Dot(normal, view) < 0.0f)
        {
            normal = -normal;
        }

        // GGX wants roughness squared; clamped away from zero so the denominator stays finite on a
        // perfect mirror.
        var alpha = ShaderMath.Max(roughness.Value * roughness.Value, 0.002f);
        var alphaSquared = alpha * alpha;
        var normalDotView = ShaderMath.Max(ShaderMath.Dot(normal, view), 1e-4f);

        var direct = new float3(0.0f, 0.0f, 0.0f);
        var specular = new float3(0.0f, 0.0f, 0.0f);
        var transmitted = new float3(0.0f, 0.0f, 0.0f);
        for (var index = 0; index < lightCount.Value; index++)
        {
            var light = lights[index];
            var toLight = light.Direction;
            var illumination = ShaderMath.Min(light.Energy, 4.0f);

            // Point, spot and area lights all fall off with distance; only a sun keeps the stored
            // direction and its raw energy.
            if (light.Kind != 2)
            {
                var delta = light.Position - input.WorldPosition;
                var distanceSquared = ShaderMath.Max(ShaderMath.Dot(delta, delta), 0.01f);
                toLight = ShaderMath.Normalize(delta);
                illumination = ShaderMath.Min(light.Energy / (12.56637f * distanceSquared), 4.0f);
            }

            // A spot cone fades between the blend-derived inner cosine and the cone edge.
            if (light.Kind == 3)
            {
                var alignment = ShaderMath.Dot(light.Direction, -toLight);
                illumination *= ShaderMath.Smoothstep(light.ConeOuter, light.ConeInner, alignment);
            }

            // An area light emits from a face rather than a point, so it only lights the side it
            // faces and spreads its energy over that face.
            if (light.Kind == 4)
            {
                illumination *= ShaderMath.Max(ShaderMath.Dot(light.Direction, -toLight), 0.0f);
            }

            var normalDotLight = ShaderMath.Dot(normal, toLight);
            var diffuse = ShaderMath.Max(normalDotLight, 0.0f);
            // A light's colour tints everything it lights, not just the highlight. Leaving it out of
            // the diffuse term made a red lamp read as white on every matte surface, which is most
            // of them, so changing the colour in Blender looked like it did nothing.
            var incident = light.Color * (diffuse * illumination);

            // Cook-Torrance GGX: normal distribution, Smith height-correlated visibility, and a
            // Schlick Fresnel grown from the material's own F0. This replaces a roughness term that
            // only scaled the highlight's brightness and never its shape.
            var halfVector = ShaderMath.Normalize(toLight + view);
            var normalDotHalf = ShaderMath.Max(ShaderMath.Dot(normal, halfVector), 0.0f);
            var viewDotHalf = ShaderMath.Max(ShaderMath.Dot(view, halfVector), 0.0f);
            var distributionDenominator =
                (normalDotHalf * normalDotHalf * (alphaSquared - 1.0f)) + 1.0f;
            var distribution =
                alphaSquared / ShaderMath.Max(
                    3.14159265f * distributionDenominator * distributionDenominator,
                    1e-6f);
            var clampedNormalDotLight = ShaderMath.Max(normalDotLight, 0.0f);
            var visibilityView =
                normalDotView + ShaderMath.Sqrt(
                    alphaSquared + ((1.0f - alphaSquared) * normalDotView * normalDotView));
            var visibilityLight =
                clampedNormalDotLight + ShaderMath.Sqrt(
                    alphaSquared
                        + ((1.0f - alphaSquared) * clampedNormalDotLight * clampedNormalDotLight));
            var visibility = 1.0f / ShaderMath.Max(
                visibilityView * visibilityLight,
                1e-6f);
            var fresnelWeight = ShaderMath.Pow(1.0f - viewDotHalf, 5.0f);
            var fresnel = f0 + ((new float3(1.0f, 1.0f, 1.0f) - f0) * fresnelWeight);
            var specularStrength = distribution * visibility * clampedNormalDotLight * illumination;
            specular += fresnel * light.Color * specularStrength;

            // Oren-Nayar qualitative model. Diffuse Roughness turns the Lambertian term into a
            // retroreflective one, which is what makes a rough dielectric read as chalk or cloth
            // rather than smooth plastic.
            var diffuseSigma = diffuseRoughness.Value * diffuseRoughness.Value;
            var orenA = 1.0f - (0.5f * diffuseSigma / (diffuseSigma + 0.33f));
            var orenB = 0.45f * diffuseSigma / (diffuseSigma + 0.09f);
            // The azimuthal term, expressed without trigonometry: the projections of the light and
            // view directions onto the surface plane, correlated.
            var lightTangent = toLight - (normal * normalDotLight);
            var viewTangent = view - (normal * ShaderMath.Dot(normal, view));
            var tangentCorrelation = ShaderMath.Max(
                ShaderMath.Dot(
                    ShaderMath.Normalize(lightTangent + new float3(1e-6f, 0.0f, 0.0f)),
                    ShaderMath.Normalize(viewTangent + new float3(1e-6f, 0.0f, 0.0f))),
                0.0f);
            var sinLight = ShaderMath.Sqrt(
                ShaderMath.Max(1.0f - (clampedNormalDotLight * clampedNormalDotLight), 0.0f));
            var sinView = ShaderMath.Sqrt(
                ShaderMath.Max(1.0f - (normalDotView * normalDotView), 0.0f));
            var orenTerm = orenA + (orenB * tangentCorrelation * ShaderMath.Max(sinLight, sinView));

            // Energy that is not reflected at the interface enters the surface: a dielectric's
            // diffuse lobe. Transmission moves that share out of the diffuse lobe and into light
            // carried through the surface, so raising it darkens the lit side rather than brightening
            // it.
            var diffuseFresnel = 1.0f - (f0.X + ((1.0f - f0.X) * fresnelWeight));
            var opaqueShare = (1.0f - transmissionWeight.Value) * dielectric * diffuseFresnel;
            direct += surface * incident * (orenTerm * opaqueShare);

            // A thin-slab approximation of what passes through: light reaching the far side is tinted
            // by the surface and lit by how squarely the light faces it. Without a second interface
            // to refract against this cannot bend anything, so it reads as translucency rather than
            // true glass -- what a single raster pass can honestly claim.
            if (transmissionWeight.Value > 0.0f)
            {
                var throughput = ShaderMath.Max(-normalDotLight, 0.0f)
                    + (clampedNormalDotLight * 0.25f);
                transmitted += surface * light.Color
                    * (throughput * illumination * transmissionWeight.Value * dielectric);
            }
        }

        // Ambient stands in for every bounce this pass does not trace, so it follows the same
        // reflect-or-absorb split as the direct term and a transmissive surface keeps less of it.
        var ambientShare = (1.0f - (transmissionWeight.Value * 0.5f)) * dielectric;
        var ambient = surface * ((0.12f + (roughness.Value * 0.08f)) * ambientShare);
        var emitted = new float3(emission.Value.R, emission.Value.G, emission.Value.B);
        var result = (ambient + direct + specular + transmitted + emitted) * exposure.Value;
        return new float4(result, 1.0f);
    }
}
