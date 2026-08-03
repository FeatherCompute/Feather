using Feather;
using Feather.Graphics;
using Feather.Math;
using Feather.RenderGraph;
using Feather.Resources;
using Feather.Shaders;

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

        var color = context.GetOrCreateRenderTarget<Rgba8, Rgba8>(Color, PixelFormat.Rgba8);

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
        var resources = context.GetOrCreateSceneResource(
            "MinimalRasterPass.SceneResources.v3",
            () => MinimalRasterSceneResources.Create(
                geometry,
                materials,
                textures,
                lights,
                includeMaterialDraws: ViewMode == 2));
        var depth = context.GetOrCreateDepthTarget(Color);
        var pipeline = resources.GetPipeline(context.SampleCount);
        IGpuTexture2D[] targets = [color];
        var cameraPosition = camera.WorldPosition;

        if (ViewMode != 2)
        {
            DrawIndexed(
                pipeline,
                resources,
                camera,
                cameraPosition,
                targets,
                depth,
                resources.FullIndices,
                resources.WhiteTexture,
                resources.FallbackInstructionBuffer,
                resources.FallbackParameterBuffer,
                resources.FallbackOutputBuffer,
                expressionInstructionCount: 0,
                baseColor: ViewMode == 0
                    ? new float4(0.8f, 0.8f, 0.8f, 1.0f)
                    : new float4(1.0f, 1.0f, 1.0f, 1.0f),
                metallic: 0.0f,
                roughness: 1.0f,
                ior: SceneMaterial.DefaultIor,
                diffuseRoughness: 0.0f,
                transmissionWeight: 0.0f,
                sheenWeight: 0.0f,
                sheenColor: SceneMaterial.DefaultSheenColor,
                clearcoatWeight: 0.0f,
                clearcoatRoughness: SceneMaterial.DefaultClearcoatRoughness,
                emission: new float4(0.0f, 0.0f, 0.0f, 1.0f),
                firstDraw: true);
            return pipeline.LastDispatchPath;
        }

        var firstDraw = true;
        foreach (var draw in resources.MaterialDraws)
        {
            var material = materials.Materials.Span[draw.MaterialIndex];
            DrawIndexed(
                pipeline,
                resources,
                camera,
                cameraPosition,
                targets,
                depth,
                draw.Indices,
                draw.Texture,
                draw.InstructionBuffer,
                draw.ParameterBuffer,
                draw.OutputBuffer,
                draw.ExpressionInstructionCount,
                material.BaseColor,
                material.Metallic,
                material.Roughness,
                material.Ior,
                material.DiffuseRoughness,
                material.TransmissionWeight,
                material.SheenWeight,
                material.SheenColor,
                material.ClearcoatWeight,
                material.ClearcoatRoughness,
                material.EmissionColor,
                firstDraw);
            firstDraw = false;
        }
        return pipeline.LastDispatchPath;
    }

    private void DrawIndexed(
        GpuGraphicsPipeline<MinimalRasterVertexShader, MinimalRasterFragmentShader, MinimalRasterVaryings> pipeline,
        MinimalRasterSceneResources resources,
        RenderCamera camera,
        float3 cameraPosition,
        IGpuTexture2D[] targets,
        GpuTexture2D<float, float> depth,
        GpuBuffer<uint> indices,
        GpuTexture2D<Rgba8, float4> texture,
        GpuBuffer<RasterMaterialInstruction> instructionBuffer,
        GpuBuffer<float4> parameterBuffer,
        GpuBuffer<MaterialExpressionOutputs> outputBuffer,
        int expressionInstructionCount,
        float4 baseColor,
        float metallic,
        float roughness,
        float ior,
        float diffuseRoughness,
        float transmissionWeight,
        float sheenWeight,
        float4 sheenColor,
        float clearcoatWeight,
        float clearcoatRoughness,
        float4 emission,
        bool firstDraw)
    {
        pipeline.DrawIndexed(
            new MinimalRasterVertexShader(
                resources.Vertices.AsReadOnly(),
                new Uniform<float4x4>(camera.ViewProjection)),
            new MinimalRasterFragmentShader(
                texture.AsSampled(),
                resources.Sampler,
                new Uniform<float4>(baseColor),
                new Uniform<float>(metallic),
                new Uniform<float>(roughness),
                new Uniform<float>(ior),
                new Uniform<float>(diffuseRoughness),
                new Uniform<float>(transmissionWeight),
                new Uniform<float>(sheenWeight),
                new Uniform<float4>(sheenColor),
                new Uniform<float>(clearcoatWeight),
                new Uniform<float>(clearcoatRoughness),
                new Uniform<float4>(emission),
                new Uniform<float>(Exposure),
                new Uniform<int>(ViewMode),
                new Uniform<int>(expressionInstructionCount),
                new Uniform<int>(resources.LightCount),
                new Uniform<float3>(cameraPosition),
                instructionBuffer.AsReadOnly(),
                parameterBuffer.AsReadOnly(),
                outputBuffer.AsReadOnly(),
                resources.LightBuffer.AsReadOnly()),
            targets,
            depth,
            indices,
            new GraphicsDrawDesc
            {
                ColorLoadOp = firstDraw ? GraphicsColorLoadOp.Clear : GraphicsColorLoadOp.Load,
                ClearColor = firstDraw ? ClearColor : null,
                DepthLoadOp = firstDraw ? GraphicsDepthLoadOp.Clear : GraphicsDepthLoadOp.Load,
                ClearDepth = firstDraw ? 1.0f : null
            });
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

    /// <summary>Scene-derived CPU conversions and GPU objects retained by the RenderHost.</summary>
    private sealed class MinimalRasterSceneResources : IDisposable
    {
        private readonly Dictionary<SampleCount,
            GpuGraphicsPipeline<MinimalRasterVertexShader, MinimalRasterFragmentShader, MinimalRasterVaryings>>
            pipelines = [];
        private readonly GpuTexture2D<Rgba8, float4>?[] sceneTextures;

        private MinimalRasterSceneResources(
            GpuBuffer<MinimalRasterVertex> vertices,
            GpuBuffer<uint> fullIndices,
            GpuBuffer<MinimalRasterLight> lightBuffer,
            int lightCount,
            SamplerState sampler,
            GpuTexture2D<Rgba8, float4> whiteTexture,
            GpuBuffer<RasterMaterialInstruction> fallbackInstructionBuffer,
            GpuBuffer<float4> fallbackParameterBuffer,
            GpuBuffer<MaterialExpressionOutputs> fallbackOutputBuffer,
            GpuTexture2D<Rgba8, float4>?[] sceneTextures,
            MinimalRasterDrawResources[] materialDraws)
        {
            Vertices = vertices;
            FullIndices = fullIndices;
            LightBuffer = lightBuffer;
            LightCount = lightCount;
            Sampler = sampler;
            WhiteTexture = whiteTexture;
            FallbackInstructionBuffer = fallbackInstructionBuffer;
            FallbackParameterBuffer = fallbackParameterBuffer;
            FallbackOutputBuffer = fallbackOutputBuffer;
            this.sceneTextures = sceneTextures;
            MaterialDraws = materialDraws;
        }

        public GpuBuffer<MinimalRasterVertex> Vertices { get; }
        public GpuBuffer<uint> FullIndices { get; }
        public GpuBuffer<MinimalRasterLight> LightBuffer { get; }
        public int LightCount { get; }
        public SamplerState Sampler { get; }
        public GpuTexture2D<Rgba8, float4> WhiteTexture { get; }
        public GpuBuffer<RasterMaterialInstruction> FallbackInstructionBuffer { get; }
        public GpuBuffer<float4> FallbackParameterBuffer { get; }
        public GpuBuffer<MaterialExpressionOutputs> FallbackOutputBuffer { get; }
        public MinimalRasterDrawResources[] MaterialDraws { get; }

        public static MinimalRasterSceneResources Create(
            SceneGeometry geometry,
            SceneMaterialTable materials,
            SceneTextureTable textures,
            SceneLightTable lights,
            bool includeMaterialDraws)
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

            var shaderLights = BuildLights(lights);
            GpuBuffer<MinimalRasterVertex>? vertices = null;
            GpuBuffer<uint>? fullIndices = null;
            GpuBuffer<MinimalRasterLight>? lightBuffer = null;
            GpuBuffer<RasterMaterialInstruction>? fallbackInstructionBuffer = null;
            GpuBuffer<float4>? fallbackParameterBuffer = null;
            GpuBuffer<MaterialExpressionOutputs>? fallbackOutputBuffer = null;
            GpuTexture2D<Rgba8, float4>? whiteTexture = null;
            var sampler = default(SamplerState);
            var samplerCreated = false;
            var sceneTextures = new GpuTexture2D<Rgba8, float4>?[textures.Textures.Length];
            var materialDraws = new List<MinimalRasterDrawResources>();
            try
            {
                vertices = GPU.CreateBuffer(shaderVertices, BufferAccess.ReadOnly);
                fullIndices = GPU.CreateIndexBuffer(geometry.Indices.Span);
                lightBuffer = GPU.CreateBuffer(shaderLights, BufferAccess.ReadOnly);
                fallbackInstructionBuffer = GPU.CreateBuffer(
                    BuildExpressionInstructions(null), BufferAccess.ReadOnly);
                fallbackParameterBuffer = GPU.CreateBuffer([float4.Zero], BufferAccess.ReadOnly);
                fallbackOutputBuffer = GPU.CreateBuffer(
                    new[] { BuildExpressionOutputs(null) }, BufferAccess.ReadOnly);
                sampler = GPU.CreateSampler(SamplerDesc.LinearRepeat);
                samplerCreated = true;
                whiteTexture = CreateWhiteTexture();

                if (includeMaterialDraws)
                {
                    var draws = geometry.Submeshes.IsEmpty
                        ? [new SceneSubmesh(0, geometry.Indices.Length, materials.DefaultMaterialIndex)]
                        : geometry.Submeshes.ToArray();
                    foreach (var draw in draws)
                    {
                        var material = materials.Materials.Span[draw.MaterialIndex];
                        var expression = material.Expression;
                        var textureIndex = expression?.TextureIndex ?? material.BaseColorTextureIndex;
                        var texture = whiteTexture;
                        if (textureIndex != SceneMaterial.NoTexture)
                        {
                            sceneTextures[textureIndex] ??=
                                CreateSceneTexture(textures.Textures.Span[textureIndex]);
                            texture = sceneTextures[textureIndex]!;
                        }
                        materialDraws.Add(MinimalRasterDrawResources.Create(
                            geometry,
                            draw,
                            expression,
                            texture));
                    }
                }

                return new MinimalRasterSceneResources(
                    vertices,
                    fullIndices,
                    lightBuffer,
                    shaderLights.Length,
                    sampler,
                    whiteTexture,
                    fallbackInstructionBuffer,
                    fallbackParameterBuffer,
                    fallbackOutputBuffer,
                    sceneTextures,
                    materialDraws.ToArray());
            }
            catch
            {
                foreach (var draw in materialDraws)
                {
                    draw.Dispose();
                }
                foreach (var texture in sceneTextures)
                {
                    texture?.Dispose();
                }
                whiteTexture?.Dispose();
                if (samplerCreated)
                {
                    sampler.Dispose();
                }
                fallbackOutputBuffer?.Dispose();
                fallbackParameterBuffer?.Dispose();
                fallbackInstructionBuffer?.Dispose();
                lightBuffer?.Dispose();
                fullIndices?.Dispose();
                vertices?.Dispose();
                throw;
            }
        }

        public GpuGraphicsPipeline<
            MinimalRasterVertexShader,
            MinimalRasterFragmentShader,
            MinimalRasterVaryings> GetPipeline(SampleCount sampleCount)
        {
            if (pipelines.TryGetValue(sampleCount, out var pipeline))
            {
                return pipeline;
            }
            pipeline = GPU.CreateGraphicsPipeline<
                MinimalRasterVertexShader,
                MinimalRasterFragmentShader,
                MinimalRasterVaryings>(
                new GraphicsPipelineDesc
                {
                    SampleCount = sampleCount,
                    DepthStencil = DepthStencilState.Default with
                    {
                        DepthTest = true,
                        DepthWrite = true,
                        DepthCompare = CompareOp.Less
                    },
                    Raster = RasterState.Default with { CullMode = CullMode.None },
                    DebugName = "Project Minimal Raster"
                });
            pipelines.Add(sampleCount, pipeline);
            return pipeline;
        }

        public void Dispose()
        {
            foreach (var pipeline in pipelines.Values)
            {
                pipeline.Dispose();
            }
            foreach (var draw in MaterialDraws)
            {
                draw.Dispose();
            }
            foreach (var texture in sceneTextures)
            {
                texture?.Dispose();
            }
            WhiteTexture.Dispose();
            Sampler.Dispose();
            FallbackOutputBuffer.Dispose();
            FallbackParameterBuffer.Dispose();
            FallbackInstructionBuffer.Dispose();
            LightBuffer.Dispose();
            FullIndices.Dispose();
            Vertices.Dispose();
        }
    }

    private sealed class MinimalRasterDrawResources : IDisposable
    {
        private MinimalRasterDrawResources(
            int materialIndex,
            GpuBuffer<uint> indices,
            GpuTexture2D<Rgba8, float4> texture,
            GpuBuffer<RasterMaterialInstruction> instructionBuffer,
            GpuBuffer<float4> parameterBuffer,
            GpuBuffer<MaterialExpressionOutputs> outputBuffer,
            int expressionInstructionCount)
        {
            MaterialIndex = materialIndex;
            Indices = indices;
            Texture = texture;
            InstructionBuffer = instructionBuffer;
            ParameterBuffer = parameterBuffer;
            OutputBuffer = outputBuffer;
            ExpressionInstructionCount = expressionInstructionCount;
        }

        public int MaterialIndex { get; }
        public GpuBuffer<uint> Indices { get; }
        public GpuTexture2D<Rgba8, float4> Texture { get; }
        public GpuBuffer<RasterMaterialInstruction> InstructionBuffer { get; }
        public GpuBuffer<float4> ParameterBuffer { get; }
        public GpuBuffer<MaterialExpressionOutputs> OutputBuffer { get; }
        public int ExpressionInstructionCount { get; }

        public static MinimalRasterDrawResources Create(
            SceneGeometry geometry,
            SceneSubmesh draw,
            SceneMaterialExpression? expression,
            GpuTexture2D<Rgba8, float4> texture)
        {
            GpuBuffer<uint>? indices = null;
            GpuBuffer<RasterMaterialInstruction>? instructionBuffer = null;
            GpuBuffer<float4>? parameterBuffer = null;
            GpuBuffer<MaterialExpressionOutputs>? outputBuffer = null;
            try
            {
                indices = GPU.CreateIndexBuffer(
                    geometry.Indices.Span.Slice(draw.FirstIndex, draw.IndexCount));
                instructionBuffer = GPU.CreateBuffer(
                    BuildExpressionInstructions(expression), BufferAccess.ReadOnly);
                parameterBuffer = GPU.CreateBuffer(
                    expression is { Parameters.IsEmpty: false }
                        ? expression.Parameters.ToArray()
                        : [float4.Zero],
                    BufferAccess.ReadOnly);
                outputBuffer = GPU.CreateBuffer(
                    new[] { BuildExpressionOutputs(expression) }, BufferAccess.ReadOnly);
                return new MinimalRasterDrawResources(
                    draw.MaterialIndex,
                    indices,
                    texture,
                    instructionBuffer,
                    parameterBuffer,
                    outputBuffer,
                    expression?.Instructions.Length ?? 0);
            }
            catch
            {
                outputBuffer?.Dispose();
                parameterBuffer?.Dispose();
                instructionBuffer?.Dispose();
                indices?.Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            OutputBuffer.Dispose();
            ParameterBuffer.Dispose();
            InstructionBuffer.Dispose();
            Indices.Dispose();
        }
    }

    private static RasterMaterialInstruction[] BuildExpressionInstructions(
        SceneMaterialExpression? expression)
    {
        if (expression is null)
        {
            return [new RasterMaterialInstruction()];
        }
        var source = expression.Instructions.Span;
        var result = new RasterMaterialInstruction[source.Length];
        for (var index = 0; index < source.Length; index++)
        {
            var item = source[index];
            result[index] = new RasterMaterialInstruction
            {
                Value = item.Value,
                Parameters = item.Parameters,
                Op = item.Op,
                A = item.A,
                B = item.B,
                C = item.C,
                D = item.D,
                E = item.E,
                F = item.F,
                G = item.G,
                H = item.H,
                ParameterOffset = item.ParameterOffset,
                ParameterCount = item.ParameterCount,
                Reserved = item.Reserved
            };
        }
        return result;
    }

    private static MaterialExpressionOutputs BuildExpressionOutputs(SceneMaterialExpression? expression)
    {
        if (expression is null)
        {
            return new MaterialExpressionOutputs();
        }
        var item = expression.Outputs;
        return new MaterialExpressionOutputs
        {
            BaseColor = item.BaseColor,
            Metallic = item.Metallic,
            Roughness = item.Roughness,
            Ior = item.Ior,
            DiffuseRoughness = item.DiffuseRoughness,
            TransmissionWeight = item.TransmissionWeight,
            SheenWeight = item.SheenWeight,
            SheenColor = item.SheenColor,
            ClearcoatWeight = item.ClearcoatWeight,
            ClearcoatRoughness = item.ClearcoatRoughness,
            EmissionColor = item.EmissionColor,
            EmissionStrength = item.EmissionStrength,
            Alpha = item.Alpha,
            Normal = item.Normal
        };
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
                // A sun needs the direction back toward its source, while a spot and an area lamp need
                // the direction they actually emit along: the cone test and the one-sided facing term
                // both negate this field themselves. The exporter already stores local -Z, so flipping
                // every kind aimed a ceiling area light away from the room and zeroed its light.
                Direction = light.Type is SceneLightType.Area or SceneLightType.Spot
                    ? light.Direction
                    : -light.Direction,
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
    Uniform<float> sheenWeight,
    Uniform<float4> sheenColor,
    Uniform<float> clearcoatWeight,
    Uniform<float> clearcoatRoughness,
    Uniform<float4> emission,
    Uniform<float> exposure,
    Uniform<int> viewMode,
    Uniform<int> expressionInstructionCount,
    Uniform<int> lightCount,
    Uniform<float3> cameraPosition,
    ReadOnlyBuffer<RasterMaterialInstruction> expressionInstructions,
    ReadOnlyBuffer<float4> expressionParameters,
    ReadOnlyBuffer<MaterialExpressionOutputs> expressionOutputs,
    // The light buffer is declared last so it lands on a high binding: the native bridge infers the
    // vertex stream from the lowest bound buffer and dedupes cross-stage buffers by source binding.
    ReadOnlyBuffer<MinimalRasterLight> lights) : IFragmentShader<MinimalRasterVaryings>
{
    public float4 Execute(MinimalRasterVaryings input)
    {
        var normal = ShaderMath.Normalize(input.Normal);
        var view = ShaderMath.Normalize(cameraPosition.Value - input.WorldPosition);
        RasterMaterialRegisters registers = default;
        for (var instructionIndex = 0;
             instructionIndex < expressionInstructionCount.Value;
             instructionIndex++)
        {
            var instruction = expressionInstructions[instructionIndex];
            var evaluated = instruction.Value;
            if (instruction.Op == 2 || instruction.Op == 3)
            {
                var coordinate = FeatherMaterialExpression.Get(registers, instruction.A).XY;
                var image = baseColorTexture.Sample(sampler, coordinate);
                evaluated = instruction.Op == 3 ? new float4(image.W) : image;
            }
            else if (instruction.Op == 9)
            {
                var factor = FeatherMaterialExpression.Get(registers, instruction.A).X;
                evaluated = expressionParameters[instruction.ParameterOffset];
                for (var element = 1; element < instruction.ParameterCount; element++)
                {
                    var p0 = expressionParameters[instruction.ParameterOffset + ((element - 1) * 2) + 1].X;
                    var p1 = expressionParameters[instruction.ParameterOffset + (element * 2) + 1].X;
                    var c0 = expressionParameters[instruction.ParameterOffset + ((element - 1) * 2)];
                    var c1 = expressionParameters[instruction.ParameterOffset + (element * 2)];
                    if (factor >= p1) evaluated = c1;
                    else if (factor >= p0 && instruction.Parameters.X > 0.5f)
                    {
                        var ramp = ShaderMath.Saturate((factor - p0) / ShaderMath.Max(p1 - p0, 1e-6f));
                        if (instruction.Parameters.X == 2.0f) ramp = ramp * ramp * (3.0f - (2.0f * ramp));
                        evaluated = ShaderMath.Lerp(c0, c1, ramp);
                    }
                }
                if (instruction.Parameters.Y > 0.5f) evaluated = new float4(evaluated.W);
            }
            else
            {
                evaluated = FeatherMaterialExpression.Evaluate(
                    instruction, registers, input.UV, normal, view);
            }
            registers = FeatherMaterialExpression.Set(registers, instructionIndex, evaluated);
        }

        var evaluatedBaseColor = baseColor.Value;
        var evaluatedMetallic = metallic.Value;
        var evaluatedRoughness = roughness.Value;
        var evaluatedIor = ior.Value;
        var evaluatedDiffuseRoughness = diffuseRoughness.Value;
        var evaluatedTransmissionWeight = transmissionWeight.Value;
        var evaluatedSheenWeight = sheenWeight.Value;
        var evaluatedSheenColor = sheenColor.Value;
        var evaluatedClearcoatWeight = clearcoatWeight.Value;
        var evaluatedClearcoatRoughness = clearcoatRoughness.Value;
        var evaluatedEmission = emission.Value;
        if (expressionInstructionCount.Value > 0)
        {
            var outputs = expressionOutputs[0];
            evaluatedBaseColor = FeatherMaterialExpression.Get(registers, outputs.BaseColor);
            evaluatedMetallic = ShaderMath.Saturate(FeatherMaterialExpression.Get(registers, outputs.Metallic).X);
            evaluatedRoughness = ShaderMath.Saturate(FeatherMaterialExpression.Get(registers, outputs.Roughness).X);
            evaluatedIor = ShaderMath.Max(FeatherMaterialExpression.Get(registers, outputs.Ior).X, 1.0f);
            evaluatedDiffuseRoughness = ShaderMath.Saturate(
                FeatherMaterialExpression.Get(registers, outputs.DiffuseRoughness).X);
            evaluatedTransmissionWeight = ShaderMath.Saturate(
                FeatherMaterialExpression.Get(registers, outputs.TransmissionWeight).X);
            evaluatedSheenWeight = ShaderMath.Saturate(FeatherMaterialExpression.Get(registers, outputs.SheenWeight).X);
            evaluatedSheenColor = FeatherMaterialExpression.Get(registers, outputs.SheenColor);
            evaluatedClearcoatWeight = ShaderMath.Saturate(
                FeatherMaterialExpression.Get(registers, outputs.ClearcoatWeight).X);
            evaluatedClearcoatRoughness = ShaderMath.Saturate(
                FeatherMaterialExpression.Get(registers, outputs.ClearcoatRoughness).X);
            var expressionEmission = FeatherMaterialExpression.Get(registers, outputs.EmissionColor);
            var expressionEmissionStrength = FeatherMaterialExpression.Get(registers, outputs.EmissionStrength).X;
            evaluatedEmission = new float4(
                expressionEmission.X * expressionEmissionStrength,
                expressionEmission.Y * expressionEmissionStrength,
                expressionEmission.Z * expressionEmissionStrength,
                expressionEmission.W);
            var tangentNormal = FeatherMaterialExpression.Get(registers, outputs.Normal).XYZ;
            if (ShaderMath.Dot(tangentNormal, tangentNormal) > 1e-6f)
            {
                normal = TangentToWorld(
                    tangentNormal, input.WorldPosition, input.UV, normal);
            }
        }

        if (viewMode.Value == 1)
        {
            return new float4((normal * 0.5f) + new float3(0.5f, 0.5f, 0.5f), 1.0f);
        }

        var sampled = expressionInstructionCount.Value > 0
            ? new float4(1.0f, 1.0f, 1.0f, 1.0f)
            : baseColorTexture.Sample(sampler, input.UV);
        var surface = new float3(
            sampled.R * evaluatedBaseColor.R,
            sampled.G * evaluatedBaseColor.G,
            sampled.B * evaluatedBaseColor.B);
        var dielectric = 1.0f - evaluatedMetallic;

        // Normal-incidence reflectance from the index of refraction, via Schlick's parameterisation
        // of Fresnel: F0 = ((n - 1) / (n + 1))^2. The old code hardcoded 0.04, which is glass at
        // n = 1.5 -- correct for exactly one material and the reason the IOR slider did nothing.
        var iorRatio = (evaluatedIor - 1.0f) / (evaluatedIor + 1.0f);
        var dielectricF0 = iorRatio * iorRatio;
        // A metal's reflectance is its base colour; a dielectric's is a colourless few percent.
        var f0 = ShaderMath.Lerp(
            new float3(dielectricF0, dielectricF0, dielectricF0),
            surface,
            evaluatedMetallic);

        // Two-sided shading: a back face lit from behind should not go black, and transmission needs
        // the geometric side to stay meaningful.
        if (ShaderMath.Dot(normal, view) < 0.0f)
        {
            normal = -normal;
        }

        // GGX wants roughness squared; clamped away from zero so the denominator stays finite on a
        // perfect mirror.
        var alpha = ShaderMath.Max(evaluatedRoughness * evaluatedRoughness, 0.002f);
        var alphaSquared = alpha * alpha;
        var normalDotView = ShaderMath.Max(ShaderMath.Dot(normal, view), 1e-4f);

        var direct = new float3(0.0f, 0.0f, 0.0f);
        var specular = new float3(0.0f, 0.0f, 0.0f);
        var clearcoat = new float3(0.0f, 0.0f, 0.0f);
        var sheen = new float3(0.0f, 0.0f, 0.0f);
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

            // Coat is a separate dielectric GGX lobe over the base surface. Sheen is deliberately
            // grazing-weighted, which keeps its coloured fabric-like response distinct from metal.
            var coatAlpha = ShaderMath.Max(
                evaluatedClearcoatRoughness * evaluatedClearcoatRoughness,
                0.002f);
            var coatAlphaSquared = coatAlpha * coatAlpha;
            var coatDenominator =
                (normalDotHalf * normalDotHalf * (coatAlphaSquared - 1.0f)) + 1.0f;
            var coatDistribution = coatAlphaSquared / ShaderMath.Max(
                3.14159265f * coatDenominator * coatDenominator,
                1e-6f);
            var coatFresnel = 0.04f + (0.96f * fresnelWeight);
            clearcoat += light.Color * (coatDistribution * visibility * clampedNormalDotLight
                * illumination * coatFresnel * evaluatedClearcoatWeight);
            var sheenFresnel = ShaderMath.Pow(1.0f - normalDotView, 5.0f);
            sheen += new float3(evaluatedSheenColor.R, evaluatedSheenColor.G, evaluatedSheenColor.B)
                * light.Color * (diffuse * illumination * sheenFresnel * evaluatedSheenWeight);

            // Oren-Nayar qualitative model. Diffuse Roughness turns the Lambertian term into a
            // retroreflective one, which is what makes a rough dielectric read as chalk or cloth
            // rather than smooth plastic.
            var diffuseSigma = evaluatedDiffuseRoughness * evaluatedDiffuseRoughness;
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
            var opaqueShare = (1.0f - evaluatedTransmissionWeight) * dielectric * diffuseFresnel;
            direct += surface * incident * (orenTerm * opaqueShare);

            // A thin-slab approximation of what passes through: light reaching the far side is tinted
            // by the surface and lit by how squarely the light faces it. Without a second interface
            // to refract against this cannot bend anything, so it reads as translucency rather than
            // true glass -- what a single raster pass can honestly claim.
            if (evaluatedTransmissionWeight > 0.0f)
            {
                var throughput = ShaderMath.Max(-normalDotLight, 0.0f)
                    + (clampedNormalDotLight * 0.25f);
                transmitted += surface * light.Color
                    * (throughput * illumination * evaluatedTransmissionWeight * dielectric);
            }
        }

        // Ambient stands in for every bounce this pass does not trace, so it follows the same
        // reflect-or-absorb split as the direct term and a transmissive surface keeps less of it.
        var ambientShare = (1.0f - (evaluatedTransmissionWeight * 0.5f)) * dielectric;
        var ambient = surface * ((0.12f + (evaluatedRoughness * 0.08f)) * ambientShare);
        var emitted = new float3(evaluatedEmission.R, evaluatedEmission.G, evaluatedEmission.B);
        var result = (ambient + direct + specular + clearcoat + sheen + transmitted + emitted) * exposure.Value;
        return new float4(result, 1.0f);
    }

    private static float4 EvaluateInstruction(
        MaterialExpressionInstruction instruction,
        MaterialExpressionRegisters registers,
        MinimalRasterVaryings input,
        float3 geometricNormal,
        float3 view,
        SampledTexture2D<float4> texture,
        SamplerState textureSampler,
        ReadOnlyBuffer<float4> parameters)
    {
        var a = GetRegister(registers, instruction.A);
        var b = GetRegister(registers, instruction.B);
        var c = GetRegister(registers, instruction.C);
        var d = GetRegister(registers, instruction.D);
        var e = GetRegister(registers, instruction.E);
        var f = GetRegister(registers, instruction.F);
        var result = instruction.Value;
        if (instruction.Op == 1)
        {
            result = new float4(input.UV, 0.0f, 1.0f);
        }
        else if (instruction.Op == 2 || instruction.Op == 3)
        {
            var image = texture.Sample(textureSampler, a.XY);
            result = instruction.Op == 3
                ? new float4(image.W, image.W, image.W, image.W)
                : image;
        }
        else if (instruction.Op == 4)
        {
            var position = a.XYZ * b.X;
            var octaves = ShaderMath.Min(c.X, 8.0f);
            var amplitude = 1.0f;
            var frequency = 1.0f;
            var total = 0.0f;
            var weight = 0.0f;
            for (var octave = 0; octave < 8; octave++)
            {
                if ((float)octave <= octaves)
                {
                    total += ValueNoise(position * frequency) * amplitude;
                    weight += amplitude;
                    amplitude *= ShaderMath.Saturate(d.X);
                    frequency *= ShaderMath.Max(e.X, 1.0f);
                }
            }
            var noise = total / ShaderMath.Max(weight, 1e-5f);
            noise = ShaderMath.Saturate(noise + (f.X * (ValueNoise(position + new float3(9.2f, 3.7f, 5.1f)) - 0.5f)));
            result = instruction.Parameters.X > 0.5f
                ? new float4(
                    noise,
                    ShaderMath.Saturate(ValueNoise(position + new float3(17.0f, 0.0f, 3.0f))),
                    ShaderMath.Saturate(ValueNoise(position + new float3(0.0f, 11.0f, 7.0f))),
                    1.0f)
                : new float4(noise, noise, noise, 1.0f);
        }
        else if (instruction.Op == 5)
        {
            var position = a.XYZ * b.X;
            var cell = ShaderMath.Floor(position);
            var nearest = 100000.0f;
            var nearestCell = float3.Zero;
            for (var z = -1; z <= 1; z++)
            {
                for (var y = -1; y <= 1; y++)
                {
                    for (var x = -1; x <= 1; x++)
                    {
                        var candidateCell = cell + new float3((float)x, (float)y, (float)z);
                        var jitter = new float3(
                            Hash(candidateCell),
                            Hash(candidateCell + new float3(19.0f, 7.0f, 3.0f)),
                            Hash(candidateCell + new float3(5.0f, 23.0f, 11.0f))) * c.X;
                        var delta = (candidateCell + jitter) - position;
                        var distance = ShaderMath.Length(delta);
                        if (distance < nearest)
                        {
                            nearest = distance;
                            nearestCell = candidateCell;
                        }
                    }
                }
            }
            result = instruction.Parameters.X > 0.5f
                ? new float4(
                    Hash(nearestCell),
                    Hash(nearestCell + new float3(13.0f, 2.0f, 17.0f)),
                    Hash(nearestCell + new float3(3.0f, 29.0f, 5.0f)),
                    1.0f)
                : new float4(nearest, nearest, nearest, 1.0f);
        }
        else if (instruction.Op == 6)
        {
            var gradient = a.X;
            if (instruction.Parameters.X == 1.0f)
            {
                gradient = a.X * a.X;
            }
            else if (instruction.Parameters.X == 2.0f)
            {
                gradient = ShaderMath.Smoothstep(0.0f, 1.0f, a.X);
            }
            else if (instruction.Parameters.X == 3.0f)
            {
                gradient = (a.X + a.Y) * 0.5f;
            }
            else if (instruction.Parameters.X == 4.0f)
            {
                gradient = ShaderMath.Length(a.XYZ);
            }
            else if (instruction.Parameters.X == 5.0f)
            {
                var radius = ShaderMath.Length(a.XYZ);
                gradient = radius * radius;
            }
            gradient = ShaderMath.Saturate(gradient);
            result = new float4(gradient, gradient, gradient, 1.0f);
        }
        else if (instruction.Op == 7)
        {
            var tile = ShaderMath.Floor(a.X * d.X) + ShaderMath.Floor(a.Y * d.X) + ShaderMath.Floor(a.Z * d.X);
            var factor = ShaderMath.Fract(tile * 0.5f) < 0.25f ? 0.0f : 1.0f;
            result = instruction.Parameters.X > 0.5f
                ? new float4(factor, factor, factor, 1.0f)
                : ShaderMath.Lerp(b, c, factor);
        }
        else if (instruction.Op == 8 || instruction.Op == 22)
        {
            var factor = instruction.Op == 22 || instruction.Parameters.X > 0.5f
                ? ShaderMath.Saturate(a.X)
                : a.X;
            result = ShaderMath.Lerp(b, c, factor);
            if (instruction.Op == 8 && instruction.Parameters.Y > 0.5f)
            {
                result = ShaderMath.Saturate(result);
            }
        }
        else if (instruction.Op == 9)
        {
            var factor = a.X;
            result = parameters[instruction.ParameterOffset];
            for (var element = 1; element < instruction.ParameterCount; element++)
            {
                var previousColor = parameters[instruction.ParameterOffset + ((element - 1) * 2)];
                var previousPosition = parameters[instruction.ParameterOffset + ((element - 1) * 2) + 1].X;
                var currentColor = parameters[instruction.ParameterOffset + (element * 2)];
                var currentPosition = parameters[instruction.ParameterOffset + (element * 2) + 1].X;
                if (factor >= currentPosition)
                {
                    result = currentColor;
                }
                else if (factor >= previousPosition && instruction.Parameters.X > 0.5f)
                {
                    var rampFactor = (factor - previousPosition) /
                        ShaderMath.Max(currentPosition - previousPosition, 1e-6f);
                    if (instruction.Parameters.X == 2.0f)
                    {
                        rampFactor = rampFactor * rampFactor * (3.0f - (2.0f * rampFactor));
                    }
                    result = ShaderMath.Lerp(previousColor, currentColor, rampFactor);
                }
            }
            if (instruction.Parameters.Y > 0.5f)
            {
                result = new float4(result.W, result.W, result.W, result.W);
            }
        }
        else if (instruction.Op == 10)
        {
            var curved = new float4(
                EvaluateCurve(b.X, 1, instruction, parameters),
                EvaluateCurve(b.Y, 2, instruction, parameters),
                EvaluateCurve(b.Z, 3, instruction, parameters),
                b.W);
            curved = new float4(
                EvaluateCurve(curved.X, 0, instruction, parameters),
                EvaluateCurve(curved.Y, 0, instruction, parameters),
                EvaluateCurve(curved.Z, 0, instruction, parameters),
                curved.W);
            result = ShaderMath.Lerp(b, curved, ShaderMath.Saturate(a.X));
        }
        else if (instruction.Op == 11)
        {
            var value = EvaluateMath(instruction.Parameters.X, a.X, b.X, c.X);
            if (instruction.Parameters.Y > 0.5f)
            {
                value = ShaderMath.Saturate(value);
            }
            result = new float4(value, value, value, value);
        }
        else if (instruction.Op == 12)
        {
            result = EvaluateVectorMath(instruction.Parameters.X, a, b, c, d);
        }
        else if (instruction.Op == 13)
        {
            var rangeFactor = (a.X - b.X) / ShaderMath.Max(c.X - b.X, 1e-6f);
            if (instruction.Parameters.X > 0.5f)
            {
                rangeFactor = ShaderMath.Saturate(rangeFactor);
            }
            var value = ShaderMath.Lerp(d.X, e.X, rangeFactor);
            result = new float4(value, value, value, value);
        }
        else if (instruction.Op == 14)
        {
            var blended = EvaluateMixRgb(instruction.Parameters.X, b, c);
            result = ShaderMath.Lerp(b, blended, ShaderMath.Saturate(a.X));
            if (instruction.Parameters.Y > 0.5f)
            {
                result = ShaderMath.Saturate(result);
            }
        }
        else if (instruction.Op == 15)
        {
            var hsv = RgbToHsv(a.XYZ);
            hsv = new float3(
                ShaderMath.Fract(hsv.X + c.X - 0.5f),
                ShaderMath.Max(hsv.Y * d.X, 0.0f),
                hsv.Z * e.X);
            var adjusted = new float4(HsvToRgb(hsv), a.W);
            result = ShaderMath.Lerp(a, adjusted, ShaderMath.Saturate(b.X));
        }
        else if (instruction.Op == 16)
        {
            var mapped = a.XYZ;
            if (instruction.Parameters.X <= 1.0f)
            {
                mapped = instruction.Parameters.X == 0.0f
                    ? (mapped * d.XYZ) + b.XYZ
                    : (mapped - b.XYZ) / new float3(
                        ShaderMath.Max(d.X, 1e-6f),
                        ShaderMath.Max(d.Y, 1e-6f),
                        ShaderMath.Max(d.Z, 1e-6f));
            }
            else
            {
                mapped *= d.XYZ;
            }
            mapped = RotateEuler(mapped, c.XYZ);
            if (instruction.Parameters.X == 3.0f)
            {
                mapped = ShaderMath.Normalize(mapped);
            }
            result = new float4(mapped, 1.0f);
        }
        else if (instruction.Op == 17)
        {
            var mapped = (a.XYZ * 2.0f) - new float3(1.0f, 1.0f, 1.0f);
            mapped = ShaderMath.Normalize(ShaderMath.Lerp(
                new float3(0.0f, 0.0f, 1.0f), mapped, ShaderMath.Max(b.X, 0.0f)));
            result = new float4(mapped, 0.0f);
        }
        else if (instruction.Op == 18)
        {
            var value = instruction.Parameters.X == 0.0f ? a.X :
                (instruction.Parameters.X == 1.0f ? a.Y : a.Z);
            result = new float4(value, value, value, value);
        }
        else if (instruction.Op == 19)
        {
            result = new float4(a.X, b.X, c.X, 1.0f);
        }
        else if (instruction.Op == 20)
        {
            var shadingNormal = ShaderMath.Dot(b.XYZ, b.XYZ) > 1e-6f
                ? ShaderMath.Normalize(b.XYZ)
                : geometricNormal;
            var cosine = ShaderMath.Saturate(ShaderMath.Abs(ShaderMath.Dot(shadingNormal, view)));
            var ratio = (a.X - 1.0f) / ShaderMath.Max(a.X + 1.0f, 1e-6f);
            var f0 = ratio * ratio;
            var fresnel = f0 + ((1.0f - f0) * ShaderMath.Pow(1.0f - cosine, 5.0f));
            result = new float4(fresnel, fresnel, fresnel, fresnel);
        }
        else if (instruction.Op == 21)
        {
            var shadingNormal = ShaderMath.Dot(b.XYZ, b.XYZ) > 1e-6f
                ? ShaderMath.Normalize(b.XYZ)
                : geometricNormal;
            var facing = ShaderMath.Saturate(ShaderMath.Abs(ShaderMath.Dot(shadingNormal, view)));
            var value = instruction.Parameters.X > 0.5f
                ? facing
                : ShaderMath.Pow(1.0f - facing, ShaderMath.Max(a.X, 1e-3f));
            result = new float4(value, value, value, value);
        }
        else if (instruction.Op == 23)
        {
            result = (a + b) * 0.5f;
        }
        return result;
    }

    [Callable]
    private static float Hash(float3 position)
    {
        return ShaderMath.Fract(ShaderMath.Sin(
            ShaderMath.Dot(position, new float3(127.1f, 311.7f, 74.7f))) * 43758.5453f);
    }

    [Callable]
    private static float ValueNoise(float3 position)
    {
        var cell = ShaderMath.Floor(position);
        var offset = ShaderMath.Fract(position);
        var weight = offset * offset * (new float3(3.0f) - (offset * 2.0f));
        var value = 0.0f;
        for (var corner = 0; corner < 8; corner++)
        {
            var x = (float)(corner % 2);
            var y = (float)((corner / 2) % 2);
            var z = (float)(corner / 4);
            var cornerOffset = new float3(x, y, z);
            var cornerWeight = (1.0f - ShaderMath.Abs(weight.X - x)) *
                (1.0f - ShaderMath.Abs(weight.Y - y)) *
                (1.0f - ShaderMath.Abs(weight.Z - z));
            value += Hash(cell + cornerOffset) * cornerWeight;
        }
        return value;
    }

    private static float EvaluateCurve(
        float value,
        int curve,
        MaterialExpressionInstruction instruction,
        ReadOnlyBuffer<float4> parameters)
    {
        var result = value;
        var found = false;
        var previous = float4.Zero;
        for (var pointIndex = 0; pointIndex < instruction.ParameterCount; pointIndex++)
        {
            var point = parameters[instruction.ParameterOffset + pointIndex];
            if ((int)point.Z == curve)
            {
                if (!found)
                {
                    result = point.Y;
                    previous = point;
                    found = true;
                }
                else if (value >= point.X)
                {
                    result = point.Y;
                    previous = point;
                }
                else if (value >= previous.X)
                {
                    var factor = (value - previous.X) / ShaderMath.Max(point.X - previous.X, 1e-6f);
                    result = ShaderMath.Lerp(previous.Y, point.Y, factor);
                }
            }
        }
        return result;
    }

    [Callable]
    private static float EvaluateMath(float operation, float a, float b, float c)
    {
        var result = a;
        if (operation == 0.0f) result = a + b;
        else if (operation == 1.0f) result = a - b;
        else if (operation == 2.0f) result = a * b;
        else if (operation == 3.0f) result = ShaderMath.Abs(b) < 1e-8f ? 0.0f : a / b;
        else if (operation == 4.0f) result = (a * b) + c;
        else if (operation == 5.0f) result = ShaderMath.Pow(ShaderMath.Abs(a), b);
        else if (operation == 6.0f) result = ShaderMath.Min(a, b);
        else if (operation == 7.0f) result = ShaderMath.Max(a, b);
        else if (operation == 8.0f) result = a < b ? 1.0f : 0.0f;
        else if (operation == 9.0f) result = a > b ? 1.0f : 0.0f;
        else if (operation == 10.0f) result = ShaderMath.Abs(a);
        else if (operation == 11.0f) result = ShaderMath.Sqrt(ShaderMath.Max(a, 0.0f));
        else if (operation == 12.0f) result = ShaderMath.Floor(a);
        else if (operation == 13.0f) result = ShaderMath.Ceil(a);
        else if (operation == 14.0f) result = ShaderMath.Fract(a);
        else if (operation == 15.0f) result = a - (b * ShaderMath.Floor(a / ShaderMath.Max(ShaderMath.Abs(b), 1e-8f)));
        else if (operation == 16.0f) result = ShaderMath.Sin(a);
        else if (operation == 17.0f) result = ShaderMath.Cos(a);
        else if (operation == 18.0f) result = ShaderMath.Tan(a);
        else if (operation == 19.0f) result = a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f);
        else if (operation == 20.0f) result = ShaderMath.Abs(a - b) <= c ? 1.0f : 0.0f;
        else if (operation == 21.0f)
        {
            var period = ShaderMath.Max(b * 2.0f, 1e-8f);
            var wrapped = a - (period * ShaderMath.Floor(a / period));
            result = b - ShaderMath.Abs(wrapped - b);
        }
        else if (operation == 22.0f) result = ShaderMath.Abs(b) < 1e-8f ? 0.0f : ShaderMath.Floor(a / b) * b;
        else if (operation == 23.0f)
        {
            var span = ShaderMath.Max(b - c, 1e-8f);
            result = c + (a - c - (span * ShaderMath.Floor((a - c) / span)));
        }
        return result;
    }

    [Callable]
    private static float4 EvaluateVectorMath(float operation, float4 a, float4 b, float4 c, float4 scale)
    {
        var av = a.XYZ;
        var bv = b.XYZ;
        var result = av;
        if (operation == 0.0f) result = av + bv;
        else if (operation == 1.0f) result = av - bv;
        else if (operation == 2.0f) result = av * bv;
        else if (operation == 3.0f) result = new float3(
            ShaderMath.Abs(bv.X) < 1e-8f ? 0.0f : av.X / bv.X,
            ShaderMath.Abs(bv.Y) < 1e-8f ? 0.0f : av.Y / bv.Y,
            ShaderMath.Abs(bv.Z) < 1e-8f ? 0.0f : av.Z / bv.Z);
        else if (operation == 4.0f) result = ShaderMath.Cross(av, bv);
        else if (operation == 5.0f)
        {
            var value = ShaderMath.Dot(av, bv);
            return new float4(value, value, value, value);
        }
        else if (operation == 6.0f)
        {
            var value = ShaderMath.Length(av - bv);
            return new float4(value, value, value, value);
        }
        else if (operation == 7.0f)
        {
            var value = ShaderMath.Length(av);
            return new float4(value, value, value, value);
        }
        else if (operation == 8.0f) result = av * scale.X;
        else if (operation == 9.0f) result = ShaderMath.Normalize(av);
        else if (operation == 10.0f) result = ShaderMath.Abs(av);
        else if (operation == 11.0f) result = ShaderMath.Min(av, bv);
        else if (operation == 12.0f) result = ShaderMath.Max(av, bv);
        else if (operation == 13.0f) result = ShaderMath.Floor(av);
        else if (operation == 14.0f) result = ShaderMath.Ceil(av);
        else if (operation == 15.0f) result = ShaderMath.Fract(av);
        else if (operation == 16.0f) result = new float3(
            av.X - (bv.X * ShaderMath.Floor(av.X / ShaderMath.Max(ShaderMath.Abs(bv.X), 1e-8f))),
            av.Y - (bv.Y * ShaderMath.Floor(av.Y / ShaderMath.Max(ShaderMath.Abs(bv.Y), 1e-8f))),
            av.Z - (bv.Z * ShaderMath.Floor(av.Z / ShaderMath.Max(ShaderMath.Abs(bv.Z), 1e-8f))));
        else if (operation == 17.0f) result = new float3(ShaderMath.Sin(av.X), ShaderMath.Sin(av.Y), ShaderMath.Sin(av.Z));
        else if (operation == 18.0f) result = new float3(ShaderMath.Cos(av.X), ShaderMath.Cos(av.Y), ShaderMath.Cos(av.Z));
        else if (operation == 19.0f) result = new float3(ShaderMath.Tan(av.X), ShaderMath.Tan(av.Y), ShaderMath.Tan(av.Z));
        return new float4(result, 1.0f);
    }

    [Callable]
    private static float4 EvaluateMixRgb(float operation, float4 a, float4 b)
    {
        if (operation == 1.0f) return a + b;
        if (operation == 2.0f) return a * b;
        if (operation == 3.0f) return a - b;
        if (operation == 4.0f) return new float4(1.0f) - ((new float4(1.0f) - a) * (new float4(1.0f) - b));
        if (operation == 5.0f) return new float4(
            ShaderMath.Abs(b.X) < 1e-8f ? 0.0f : a.X / b.X,
            ShaderMath.Abs(b.Y) < 1e-8f ? 0.0f : a.Y / b.Y,
            ShaderMath.Abs(b.Z) < 1e-8f ? 0.0f : a.Z / b.Z,
            a.W);
        if (operation == 6.0f) return ShaderMath.Abs(a - b);
        if (operation == 7.0f) return ShaderMath.Min(a, b);
        if (operation == 8.0f) return ShaderMath.Max(a, b);
        if (operation == 9.0f) return new float4(
            Overlay(a.X, b.X), Overlay(a.Y, b.Y), Overlay(a.Z, b.Z), a.W);
        return b;
    }

    [Callable]
    private static float Overlay(float a, float b)
    {
        return a < 0.5f ? 2.0f * a * b : 1.0f - (2.0f * (1.0f - a) * (1.0f - b));
    }

    [Callable]
    private static float3 RgbToHsv(float3 color)
    {
        var maximum = ShaderMath.Max(color.X, ShaderMath.Max(color.Y, color.Z));
        var minimum = ShaderMath.Min(color.X, ShaderMath.Min(color.Y, color.Z));
        var delta = maximum - minimum;
        var hue = 0.0f;
        if (delta > 1e-6f)
        {
            if (maximum == color.X) hue = (color.Y - color.Z) / delta;
            else if (maximum == color.Y) hue = 2.0f + ((color.Z - color.X) / delta);
            else hue = 4.0f + ((color.X - color.Y) / delta);
            hue = ShaderMath.Fract(hue / 6.0f);
        }
        var saturation = maximum <= 1e-6f ? 0.0f : delta / maximum;
        return new float3(hue, saturation, maximum);
    }

    [Callable]
    private static float3 HsvToRgb(float3 hsv)
    {
        var h = ShaderMath.Fract(hsv.X) * 6.0f;
        var sector = ShaderMath.Floor(h);
        var fraction = h - sector;
        var p = hsv.Z * (1.0f - hsv.Y);
        var q = hsv.Z * (1.0f - (hsv.Y * fraction));
        var t = hsv.Z * (1.0f - (hsv.Y * (1.0f - fraction)));
        if (sector < 1.0f) return new float3(hsv.Z, t, p);
        if (sector < 2.0f) return new float3(q, hsv.Z, p);
        if (sector < 3.0f) return new float3(p, hsv.Z, t);
        if (sector < 4.0f) return new float3(p, q, hsv.Z);
        if (sector < 5.0f) return new float3(t, p, hsv.Z);
        return new float3(hsv.Z, p, q);
    }

    [Callable]
    private static float3 RotateEuler(float3 value, float3 rotation)
    {
        var cx = ShaderMath.Cos(rotation.X);
        var sx = ShaderMath.Sin(rotation.X);
        var cy = ShaderMath.Cos(rotation.Y);
        var sy = ShaderMath.Sin(rotation.Y);
        var cz = ShaderMath.Cos(rotation.Z);
        var sz = ShaderMath.Sin(rotation.Z);
        var xRotated = new float3(value.X, (value.Y * cx) - (value.Z * sx), (value.Y * sx) + (value.Z * cx));
        var yRotated = new float3((xRotated.X * cy) + (xRotated.Z * sy), xRotated.Y, (-xRotated.X * sy) + (xRotated.Z * cy));
        return new float3((yRotated.X * cz) - (yRotated.Y * sz), (yRotated.X * sz) + (yRotated.Y * cz), yRotated.Z);
    }

    [Callable]
    private static float3 TangentToWorld(
        float3 tangentNormal,
        float3 worldPosition,
        float2 uv,
        float3 geometricNormal)
    {
        var positionDx = ShaderMath.Ddx(worldPosition);
        var positionDy = ShaderMath.Ddy(worldPosition);
        var uvDx = ShaderMath.Ddx(uv);
        var uvDy = ShaderMath.Ddy(uv);
        var determinant = (uvDx.X * uvDy.Y) - (uvDx.Y * uvDy.X);
        if (ShaderMath.Abs(determinant) < 1e-8f)
        {
            return geometricNormal;
        }
        var tangent = ShaderMath.Normalize(
            ((positionDx * uvDy.Y) - (positionDy * uvDx.Y)) / determinant);
        var bitangent = ShaderMath.Normalize(
            ((positionDy * uvDx.X) - (positionDx * uvDy.X)) / determinant);
        return ShaderMath.Normalize(
            (tangent * tangentNormal.X) +
            (bitangent * tangentNormal.Y) +
            (geometricNormal * tangentNormal.Z));
    }

    [Callable]
    private static float4 GetRegister(MaterialExpressionRegisters registers, int index)
    {
        if (index == 0) return registers.R0;
        if (index == 1) return registers.R1;
        if (index == 2) return registers.R2;
        if (index == 3) return registers.R3;
        if (index == 4) return registers.R4;
        if (index == 5) return registers.R5;
        if (index == 6) return registers.R6;
        if (index == 7) return registers.R7;
        if (index == 8) return registers.R8;
        if (index == 9) return registers.R9;
        if (index == 10) return registers.R10;
        if (index == 11) return registers.R11;
        if (index == 12) return registers.R12;
        if (index == 13) return registers.R13;
        if (index == 14) return registers.R14;
        if (index == 15) return registers.R15;
        if (index == 16) return registers.R16;
        if (index == 17) return registers.R17;
        if (index == 18) return registers.R18;
        if (index == 19) return registers.R19;
        if (index == 20) return registers.R20;
        if (index == 21) return registers.R21;
        if (index == 22) return registers.R22;
        if (index == 23) return registers.R23;
        if (index == 24) return registers.R24;
        if (index == 25) return registers.R25;
        if (index == 26) return registers.R26;
        if (index == 27) return registers.R27;
        if (index == 28) return registers.R28;
        if (index == 29) return registers.R29;
        if (index == 30) return registers.R30;
        if (index == 31) return registers.R31;
        if (index == 32) return registers.R32;
        if (index == 33) return registers.R33;
        if (index == 34) return registers.R34;
        if (index == 35) return registers.R35;
        if (index == 36) return registers.R36;
        if (index == 37) return registers.R37;
        if (index == 38) return registers.R38;
        if (index == 39) return registers.R39;
        if (index == 40) return registers.R40;
        if (index == 41) return registers.R41;
        if (index == 42) return registers.R42;
        if (index == 43) return registers.R43;
        if (index == 44) return registers.R44;
        if (index == 45) return registers.R45;
        if (index == 46) return registers.R46;
        if (index == 47) return registers.R47;
        if (index == 48) return registers.R48;
        if (index == 49) return registers.R49;
        if (index == 50) return registers.R50;
        if (index == 51) return registers.R51;
        if (index == 52) return registers.R52;
        if (index == 53) return registers.R53;
        if (index == 54) return registers.R54;
        if (index == 55) return registers.R55;
        if (index == 56) return registers.R56;
        if (index == 57) return registers.R57;
        if (index == 58) return registers.R58;
        if (index == 59) return registers.R59;
        if (index == 60) return registers.R60;
        if (index == 61) return registers.R61;
        if (index == 62) return registers.R62;
        if (index == 63) return registers.R63;
        return float4.Zero;
    }

    [Callable]
    private static MaterialExpressionRegisters SetRegister(
        MaterialExpressionRegisters registers,
        int index,
        float4 value)
    {
        if (index == 0) registers.R0 = value;
        else if (index == 1) registers.R1 = value;
        else if (index == 2) registers.R2 = value;
        else if (index == 3) registers.R3 = value;
        else if (index == 4) registers.R4 = value;
        else if (index == 5) registers.R5 = value;
        else if (index == 6) registers.R6 = value;
        else if (index == 7) registers.R7 = value;
        else if (index == 8) registers.R8 = value;
        else if (index == 9) registers.R9 = value;
        else if (index == 10) registers.R10 = value;
        else if (index == 11) registers.R11 = value;
        else if (index == 12) registers.R12 = value;
        else if (index == 13) registers.R13 = value;
        else if (index == 14) registers.R14 = value;
        else if (index == 15) registers.R15 = value;
        else if (index == 16) registers.R16 = value;
        else if (index == 17) registers.R17 = value;
        else if (index == 18) registers.R18 = value;
        else if (index == 19) registers.R19 = value;
        else if (index == 20) registers.R20 = value;
        else if (index == 21) registers.R21 = value;
        else if (index == 22) registers.R22 = value;
        else if (index == 23) registers.R23 = value;
        else if (index == 24) registers.R24 = value;
        else if (index == 25) registers.R25 = value;
        else if (index == 26) registers.R26 = value;
        else if (index == 27) registers.R27 = value;
        else if (index == 28) registers.R28 = value;
        else if (index == 29) registers.R29 = value;
        else if (index == 30) registers.R30 = value;
        else if (index == 31) registers.R31 = value;
        else if (index == 32) registers.R32 = value;
        else if (index == 33) registers.R33 = value;
        else if (index == 34) registers.R34 = value;
        else if (index == 35) registers.R35 = value;
        else if (index == 36) registers.R36 = value;
        else if (index == 37) registers.R37 = value;
        else if (index == 38) registers.R38 = value;
        else if (index == 39) registers.R39 = value;
        else if (index == 40) registers.R40 = value;
        else if (index == 41) registers.R41 = value;
        else if (index == 42) registers.R42 = value;
        else if (index == 43) registers.R43 = value;
        else if (index == 44) registers.R44 = value;
        else if (index == 45) registers.R45 = value;
        else if (index == 46) registers.R46 = value;
        else if (index == 47) registers.R47 = value;
        else if (index == 48) registers.R48 = value;
        else if (index == 49) registers.R49 = value;
        else if (index == 50) registers.R50 = value;
        else if (index == 51) registers.R51 = value;
        else if (index == 52) registers.R52 = value;
        else if (index == 53) registers.R53 = value;
        else if (index == 54) registers.R54 = value;
        else if (index == 55) registers.R55 = value;
        else if (index == 56) registers.R56 = value;
        else if (index == 57) registers.R57 = value;
        else if (index == 58) registers.R58 = value;
        else if (index == 59) registers.R59 = value;
        else if (index == 60) registers.R60 = value;
        else if (index == 61) registers.R61 = value;
        else if (index == 62) registers.R62 = value;
        else if (index == 63) registers.R63 = value;
        return registers;
    }
}

[GpuStruct]
public partial struct MaterialExpressionRegisters
{
    public float4 R0; public float4 R1; public float4 R2; public float4 R3;
    public float4 R4; public float4 R5; public float4 R6; public float4 R7;
    public float4 R8; public float4 R9; public float4 R10; public float4 R11;
    public float4 R12; public float4 R13; public float4 R14; public float4 R15;
    public float4 R16; public float4 R17; public float4 R18; public float4 R19;
    public float4 R20; public float4 R21; public float4 R22; public float4 R23;
    public float4 R24; public float4 R25; public float4 R26; public float4 R27;
    public float4 R28; public float4 R29; public float4 R30; public float4 R31;
    public float4 R32; public float4 R33; public float4 R34; public float4 R35;
    public float4 R36; public float4 R37; public float4 R38; public float4 R39;
    public float4 R40; public float4 R41; public float4 R42; public float4 R43;
    public float4 R44; public float4 R45; public float4 R46; public float4 R47;
    public float4 R48; public float4 R49; public float4 R50; public float4 R51;
    public float4 R52; public float4 R53; public float4 R54; public float4 R55;
    public float4 R56; public float4 R57; public float4 R58; public float4 R59;
    public float4 R60; public float4 R61; public float4 R62; public float4 R63;
}

[GpuStruct]
public partial struct MaterialExpressionInstruction
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

[GpuStruct]
public partial struct MaterialExpressionOutputs
{
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
