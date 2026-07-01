using System.Globalization;
using Feather;
using Feather.Graphics;
using Feather.Math;
using Feather.Resources;
using Feather.Windowing;

const int WindowWidth = 1280;
const int WindowHeight = 720;
const float FieldOfViewDegrees = 60.0f;
const float NearPlane = 0.1f;
const float FarPlane = 100.0f;
const float ModelScale = 0.01f;
const float MouseSensitivity = 0.003f;
const float MoveSpeed = 0.05f;
const int AtlasSize = 4096;
const int AtlasGrid = 8;
const int AtlasGutter = 2;
const SampleCount MsaaSamples = SampleCount.X4;

var options = SponzaOptions.Parse(args);
var baseDirectory = options.SponzaDirectory is not null ? Path.GetFullPath(options.SponzaDirectory) : FindSponzaDirectory();
var objPath = Path.Combine(baseDirectory, "sponza.obj");

Console.WriteLine("=== Feather Sponza Renderer ===");
Console.WriteLine($"Loading {objPath}");

var materials = new MaterialLibrary();
var mesh = ObjMesh.Load(objPath, baseDirectory, materials);
var atlas = TextureAtlas.Build(materials, AtlasSize, AtlasGrid, AtlasGutter);
var vertices = mesh.Flatten(materials, atlas);
var center = mesh.Center;
var radius = mesh.Radius;

Console.WriteLine($"Positions: {mesh.PositionCount:N0}  Triangles: {mesh.TriangleCount:N0}");
Console.WriteLine($"Materials: {materials.Count:N0}  Atlas textures: {atlas.LoadedTextureCount:N0}");
Console.WriteLine($"Vertices: {vertices.Length:N0}  Center: {center}  Radius: {radius:N2}");

if (options.CapturePath is not null)
{
    RenderCapture(options.CapturePath, vertices, atlas, center);
    return;
}

using var window = GpuWindow.Create(new()
{
    Width = WindowWidth,
    Height = WindowHeight,
    Title = "Feather - Sponza Atrium",
    Resizable = true,
    VSync = true
});
using var presenter = window.CreateTexturePresenter();
using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(window.Width, window.Height, PixelFormat.Rgba8);
using var depth = GPU.CreateDepthTexture2D(window.Width, window.Height);
using var scene = SponzaGpuScene.Create(vertices, atlas, MsaaSamples);

var projection = MatrixMath.PerspectiveVk(FieldOfViewDegrees, (float)window.Width / window.Height, NearPlane, FarPlane);
var model = MatrixMath.Translation(-center * ModelScale) * MatrixMath.Scale(ModelScale);
var camera = new Camera(new float3(0.0f, 2.0f, 5.0f), 0.0f, -0.2f);
var firstMouseSample = true;
var lastMouse = float2.Zero;

Console.WriteLine($"Pipeline ready. MSAA = {(uint)MsaaSamples}x. Atlas mips = {scene.AtlasMipLevels}. WASD = move, mouse = look, ESC = exit.");

while (window.IsOpen)
{
    window.PollEvents();
    while (window.TryPollEvent(out var windowEvent))
    {
        if (windowEvent is WindowKeyEvent { Key: WindowKey.Escape, Pressed: true })
        {
            window.Close();
        }
    }

    var mouse = new float2(window.MousePosition.X, window.MousePosition.Y);
    if (firstMouseSample)
    {
        lastMouse = mouse;
        firstMouseSample = false;
    }

    var delta = mouse - lastMouse;
    lastMouse = mouse;
    camera.Yaw -= delta.X * MouseSensitivity;
    camera.Pitch = Math.Clamp(camera.Pitch - (delta.Y * MouseSensitivity), -1.5f, 1.5f);

    var moveStep = MoveSpeed;
    if (window.IsKeyDown(WindowKey.LeftShift) || window.IsKeyDown(WindowKey.RightShift))
    {
        moveStep *= 4.0f;
    }

    var forward = camera.FlatForward;
    var right = camera.Right;
    if (window.IsKeyDown(WindowKey.W))
    {
        camera.Position += forward * moveStep;
    }

    if (window.IsKeyDown(WindowKey.S))
    {
        camera.Position -= forward * moveStep;
    }

    if (window.IsKeyDown(WindowKey.A))
    {
        camera.Position -= right * moveStep;
    }

    if (window.IsKeyDown(WindowKey.D))
    {
        camera.Position += right * moveStep;
    }

    var view = MatrixMath.CameraView(camera.Position, camera.Yaw, camera.Pitch);
    var mvp = projection * view * model;

    scene.Draw(color, depth, mvp);
    presenter.Present(color);
}

static void RenderCapture(string path, SponzaVertex[] vertices, TextureAtlas atlas, float3 center)
{
    using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(WindowWidth, WindowHeight, PixelFormat.Rgba8);
    using var depth = GPU.CreateDepthTexture2D(WindowWidth, WindowHeight);
    using var scene = SponzaGpuScene.Create(vertices, atlas, MsaaSamples);

    var projection = MatrixMath.PerspectiveVk(FieldOfViewDegrees, (float)WindowWidth / WindowHeight, NearPlane, FarPlane);
    var model = MatrixMath.Translation(-center * ModelScale) * MatrixMath.Scale(ModelScale);
    var camera = new Camera(new float3(0.0f, 2.0f, 5.0f), 0.0f, -0.2f);
    var mvp = projection * MatrixMath.CameraView(camera.Position, camera.Yaw, camera.Pitch) * model;

    scene.Draw(color, depth, mvp);
    var directory = Path.GetDirectoryName(Path.GetFullPath(path));
    if (!string.IsNullOrWhiteSpace(directory))
    {
        Directory.CreateDirectory(directory);
    }

    color.Save(path);
    var stats = AnalyzePixels(color);
    Console.WriteLine($"Captured {path}");
    Console.WriteLine($"Capture pixels: non-background {stats.NonBackground:N0}/{stats.Total:N0}, unique sample {stats.UniqueSampleCount:N0}, avg rgb ({stats.AverageR:F1}, {stats.AverageG:F1}, {stats.AverageB:F1})");
}

static PixelStats AnalyzePixels(GpuTexture2D<Rgba32, Rgba32> texture)
{
    var pixels = new Rgba32[texture.Width * texture.Height];
    texture.Read(pixels);
    var unique = new HashSet<Rgba32>();
    long r = 0;
    long g = 0;
    long b = 0;
    var nonBackground = 0;
    var stride = Math.Max(1, pixels.Length / 4096);
    for (var i = 0; i < pixels.Length; i++)
    {
        var pixel = pixels[i];
        r += pixel.R;
        g += pixel.G;
        b += pixel.B;
        if (pixel is not { R: 0, G: 0, B: 0, A: 0 } and not { R: 0, G: 0, B: 0, A: 255 })
        {
            nonBackground++;
        }

        if (i % stride == 0)
        {
            unique.Add(pixel);
        }
    }

    return new PixelStats(
        pixels.Length,
        nonBackground,
        unique.Count,
        (double)r / pixels.Length,
        (double)g / pixels.Length,
        (double)b / pixels.Length);
}

static string FindSponzaDirectory()
{
    var current = AppContext.BaseDirectory;
    for (var i = 0; i < 8; i++)
    {
        var candidate = Path.Combine(current, "Sponza");
        if (File.Exists(Path.Combine(candidate, "sponza.obj")))
        {
            return candidate;
        }

        current = Directory.GetParent(current)?.FullName
                  ?? throw new DirectoryNotFoundException(
                      "Could not locate Sponza/sponza.obj. Pass the Sponza directory as the first argument.");
    }

    throw new DirectoryNotFoundException(
        "Could not locate Sponza/sponza.obj. Pass the Sponza directory as the first argument.");
}

file sealed class SponzaGpuScene : IDisposable
{
    private readonly GpuBuffer<SponzaVertex> vertexBuffer;
    private readonly GpuTexture2D<Rgba32, float4> texture;
    private readonly SamplerState sampler;
    private readonly GpuGraphicsPipeline<SponzaVertexShader, SponzaFragmentShader, SponzaVaryings> pipeline;
    private readonly uint vertexCount;

    private SponzaGpuScene(
        GpuBuffer<SponzaVertex> vertexBuffer,
        GpuTexture2D<Rgba32, float4> texture,
        SamplerState sampler,
        GpuGraphicsPipeline<SponzaVertexShader, SponzaFragmentShader, SponzaVaryings> pipeline,
        int atlasMipLevels,
        uint vertexCount)
    {
        this.vertexBuffer = vertexBuffer;
        this.texture = texture;
        this.sampler = sampler;
        this.pipeline = pipeline;
        AtlasMipLevels = atlasMipLevels;
        this.vertexCount = vertexCount;
    }

    public int AtlasMipLevels { get; }

    public static SponzaGpuScene Create(SponzaVertex[] vertices, TextureAtlas atlas, SampleCount sampleCount)
    {
        var vertexBuffer = GPU.CreateBuffer<SponzaVertex>(vertices, BufferAccess.ReadOnly);
        var atlasMipLevels = CalculateMipLevelCount(atlas.Width, atlas.Height);
        var texture = GPU.CreateTexture2D<Rgba32, float4>(
            atlas.Width,
            atlas.Height,
            atlasMipLevels,
            PixelFormat.Rgba8,
            TextureAccess.Sampled);
        var sampler = GPU.CreateSampler(SamplerDesc.LinearClamp);
        var pipeline = GPU.CreateGraphicsPipeline<SponzaVertexShader, SponzaFragmentShader, SponzaVaryings>(
            new GraphicsPipelineDesc
            {
                DepthTest = true,
                DepthWrite = true,
                SampleCount = sampleCount,
                DebugName = "Feather Sponza"
            });

        texture.Upload(atlas.Pixels);
        texture.GenerateMipmaps();
        return new SponzaGpuScene(vertexBuffer, texture, sampler, pipeline, atlasMipLevels, (uint)vertices.Length);
    }

    public void Draw<TPixel, TValue, TDepthPixel, TDepthValue>(
        GpuTexture2D<TPixel, TValue> color,
        GpuTexture2D<TDepthPixel, TDepthValue> depth,
        float4x4 mvp)
        where TPixel : unmanaged
        where TValue : unmanaged
        where TDepthPixel : unmanaged
        where TDepthValue : unmanaged
    {
        pipeline.Draw(
            new SponzaVertexShader(vertexBuffer.AsReadOnly(), new Uniform<float4x4>(mvp)),
            new SponzaFragmentShader(texture.AsSampled(), sampler),
            color,
            depth,
            vertexCount,
            new GraphicsDrawDesc { DepthLoadOp = GraphicsDepthLoadOp.Clear, ClearDepth = 1.0f });
    }

    public void Dispose()
    {
        pipeline.Dispose();
        sampler.Dispose();
        texture.Dispose();
        vertexBuffer.Dispose();
    }

    private static int CalculateMipLevelCount(int width, int height)
    {
        var levels = 1;
        var size = Math.Max(width, height);
        while (size > 1)
        {
            size /= 2;
            levels++;
        }

        return levels;
    }
}

file readonly record struct SponzaOptions(string? SponzaDirectory, string? CapturePath)
{
    public static SponzaOptions Parse(string[] args)
    {
        string? directory = null;
        string? capture = null;
        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (arg == "--capture")
            {
                if (i + 1 >= args.Length)
                {
                    throw new ArgumentException("--capture requires a path.");
                }

                capture = args[++i];
                continue;
            }

            if (arg.StartsWith("--capture=", StringComparison.Ordinal))
            {
                capture = arg["--capture=".Length..];
                continue;
            }

            directory ??= arg;
        }

        return new SponzaOptions(directory, capture);
    }
}

file readonly record struct PixelStats(
    int Total,
    int NonBackground,
    int UniqueSampleCount,
    double AverageR,
    double AverageG,
    double AverageB);

public readonly record struct Rgba32(byte R, byte G, byte B, byte A);

[GpuStruct]
public partial struct SponzaVertex
{
    public float3 Position;
    public float3 Normal;
    public float2 Uv;
    public float4 AtlasTransform;
}

[GpuStruct]
public partial struct SponzaVaryings
{
    [Position] public float4 Position;
    public float3 Normal;
    public float2 Uv;
    public float4 AtlasTransform;
}

[VertexShader]
public readonly partial struct SponzaVertexShader(
    ReadOnlyBuffer<SponzaVertex> vertices,
    Uniform<float4x4> mvp) : IVertexShader<SponzaVaryings>
{
    public SponzaVaryings Execute()
    {
        var vertex = vertices[VertexIds.Index];
        return new SponzaVaryings
        {
            Position = ShaderMath.Mul(mvp.Value, new float4(vertex.Position, 1.0f)),
            Normal = ShaderMath.Normalize(vertex.Normal),
            Uv = vertex.Uv,
            AtlasTransform = vertex.AtlasTransform
        };
    }
}

[FragmentShader]
public readonly partial struct SponzaFragmentShader(
    SampledTexture2D<float4> texture,
    SamplerState sampler) : IFragmentShader<SponzaVaryings>
{
    public float4 Execute(SponzaVaryings input)
    {
        var tiled = ShaderMath.Fract(input.Uv);
        var uv = input.AtlasTransform.XY + (tiled * input.AtlasTransform.ZW);
        var ddx = ShaderMath.Ddx(input.Uv) * input.AtlasTransform.ZW;
        var ddy = ShaderMath.Ddy(input.Uv) * input.AtlasTransform.ZW;
        var normal = ShaderMath.Normalize(input.Normal);
        var light = ShaderMath.Max(ShaderMath.Dot(normal, ShaderMath.Normalize(new float3(0.3f, 0.6f, 0.4f))), 0.2f);
        var sampled = texture.SampleGrad(sampler, uv, ddx, ddy);
        return new float4(sampled.R * light, sampled.G * light, sampled.B * light, 1.0f);
    }
}

file sealed class MaterialLibrary
{
    private readonly Dictionary<string, Material> materials = new(StringComparer.Ordinal);

    public int Count => materials.Count;

    public IEnumerable<Material> All => materials.Values;

    public void Load(string path, string baseDirectory)
    {
        using var reader = File.OpenText(path);
        Material? current = null;

        while (reader.ReadLine() is { } line)
        {
            var parts = SplitTokens(line);
            if (parts.Length == 0 || parts[0].StartsWith('#'))
            {
                continue;
            }

            switch (parts[0])
            {
                case "newmtl":
                    Commit(current);
                    current = new Material(parts[1]);
                    break;
                case "Kd" when current is not null && parts.Length >= 4:
                    current.DiffuseColor = new float3(ParseFloat(parts[1]), ParseFloat(parts[2]), ParseFloat(parts[3]));
                    break;
                case "map_Kd" when current is not null && parts.Length >= 2:
                    current.DiffuseTexturePath = ResolveMaterialPath(baseDirectory, parts[^1]);
                    break;
            }
        }

        Commit(current);
    }

    public Material? Find(string name)
        => materials.TryGetValue(name, out var material) ? material : null;

    public void AssignAtlasSlots(int maxSlots)
    {
        var slot = 1;
        foreach (var material in materials.Values.OrderBy(static material => material.Name, StringComparer.Ordinal))
        {
            if (!string.IsNullOrWhiteSpace(material.DiffuseTexturePath) && File.Exists(material.DiffuseTexturePath) &&
                slot < maxSlots)
            {
                material.AtlasSlot = slot++;
                continue;
            }

            material.AtlasSlot = 0;
        }
    }

    private void Commit(Material? material)
    {
        if (material is not null && material.Name.Length > 0)
        {
            materials[material.Name] = material;
        }
    }

    private static string ResolveMaterialPath(string baseDirectory, string value)
    {
        var normalized = value.Replace('\\', Path.DirectorySeparatorChar).Replace('/', Path.DirectorySeparatorChar);
        return Path.IsPathRooted(normalized) ? normalized : Path.Combine(baseDirectory, normalized);
    }

    private static string[] SplitTokens(string line)
        => line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    private static float ParseFloat(string value)
        => float.Parse(value, CultureInfo.InvariantCulture);
}

file sealed class Material(string name)
{
    public string Name { get; } = name;
    public string? DiffuseTexturePath { get; set; }
    public float3 DiffuseColor { get; set; } = float3.One;
    public int AtlasSlot { get; set; }
}

file sealed class TextureAtlas
{
    private TextureAtlas(int width, int height, int grid, int gutter, Rgba32[] pixels, int loadedTextureCount)
    {
        Width = width;
        Height = height;
        Grid = grid;
        Gutter = gutter;
        Pixels = pixels;
        LoadedTextureCount = loadedTextureCount;
    }

    public int Width { get; }
    public int Height { get; }
    public int Grid { get; }
    public int Gutter { get; }
    public Rgba32[] Pixels { get; }
    public int LoadedTextureCount { get; }

    public static TextureAtlas Build(MaterialLibrary materials, int size, int grid, int gutter)
    {
        materials.AssignAtlasSlots(grid * grid);
        var pixels = Enumerable.Repeat(new Rgba32(128, 128, 128, 255), size * size).ToArray();
        var slotSize = size / grid;
        FillSlot(pixels, size, grid, slot: 0, new Rgba32(255, 255, 255, 255));

        var loaded = 0;
        foreach (var material in materials.All.OrderBy(static material => material.Name, StringComparer.Ordinal))
        {
            if (material.AtlasSlot <= 0 || material.DiffuseTexturePath is null)
            {
                continue;
            }

            try
            {
                var image = TgaImage.Load(material.DiffuseTexturePath);
                CopyScaledSlot(image.Pixels, image.Width, image.Height, pixels, size, grid, gutter, material.AtlasSlot);
                loaded++;
            }
            catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException)
            {
                material.AtlasSlot = 0;
                Console.WriteLine($"Skipping texture '{material.DiffuseTexturePath}': {ex.Message}");
            }
        }

        Console.WriteLine($"Atlas: {size}x{size}  Grid: {grid}x{grid}  Slot: {slotSize}px");
        return new TextureAtlas(size, size, grid, gutter, pixels, loaded);
    }

    public float4 GetUvTransform(int slot)
    {
        var slotSize = Width / Grid;
        var clampedSlot = Math.Clamp(slot, 0, (Grid * Grid) - 1);
        var slotX = clampedSlot % Grid;
        var slotY = clampedSlot / Grid;
        var offsetX = (float)((slotX * slotSize) + Gutter) / Width;
        var offsetY = (float)((slotY * slotSize) + Gutter) / Height;
        var scale = (float)(slotSize - (Gutter * 2)) / Width;
        return new float4(offsetX, offsetY, scale, scale);
    }

    private static void FillSlot(Rgba32[] destination, int atlasSize, int grid, int slot, Rgba32 color)
    {
        var slotSize = atlasSize / grid;
        var startX = (slot % grid) * slotSize;
        var startY = (slot / grid) * slotSize;
        for (var y = 0; y < slotSize; y++)
        {
            var row = (startY + y) * atlasSize;
            for (var x = 0; x < slotSize; x++)
            {
                destination[row + startX + x] = color;
            }
        }
    }

    private static void CopyScaledSlot(
        Rgba32[] source,
        int sourceWidth,
        int sourceHeight,
        Rgba32[] destination,
        int atlasSize,
        int grid,
        int gutter,
        int slot)
    {
        var slotSize = atlasSize / grid;
        var inner = slotSize - (gutter * 2);
        var startX = (slot % grid) * slotSize;
        var startY = (slot / grid) * slotSize;

        for (var y = 0; y < slotSize; y++)
        {
            for (var x = 0; x < slotSize; x++)
            {
                var sampleX = Math.Clamp(x - gutter, 0, inner - 1);
                var sampleY = Math.Clamp(y - gutter, 0, inner - 1);
                var u = inner <= 1 ? 0.0f : (float)sampleX / (inner - 1);
                var v = inner <= 1 ? 0.0f : (float)sampleY / (inner - 1);
                destination[((startY + y) * atlasSize) + startX + x] =
                    SampleBilinear(source, sourceWidth, sourceHeight, u, v);
            }
        }
    }

    private static Rgba32 SampleBilinear(Rgba32[] pixels, int width, int height, float u, float v)
    {
        var x = u * (width - 1);
        var y = v * (height - 1);
        var x0 = Math.Clamp((int)MathF.Floor(x), 0, width - 1);
        var y0 = Math.Clamp((int)MathF.Floor(y), 0, height - 1);
        var x1 = Math.Min(x0 + 1, width - 1);
        var y1 = Math.Min(y0 + 1, height - 1);
        var tx = x - x0;
        var ty = y - y0;
        var c00 = pixels[(y0 * width) + x0];
        var c10 = pixels[(y0 * width) + x1];
        var c01 = pixels[(y1 * width) + x0];
        var c11 = pixels[(y1 * width) + x1];
        return new Rgba32(
            Blend(c00.R, c10.R, c01.R, c11.R, tx, ty),
            Blend(c00.G, c10.G, c01.G, c11.G, tx, ty),
            Blend(c00.B, c10.B, c01.B, c11.B, tx, ty),
            Blend(c00.A, c10.A, c01.A, c11.A, tx, ty));
    }

    private static byte Blend(byte c00, byte c10, byte c01, byte c11, float tx, float ty)
    {
        var top = c00 + ((c10 - c00) * tx);
        var bottom = c01 + ((c11 - c01) * tx);
        return (byte)Math.Clamp(MathF.Round(top + ((bottom - top) * ty)), 0, 255);
    }
}

file sealed class ObjMesh
{
    private readonly List<float3> positions = [];
    private readonly List<float3> normals = [];
    private readonly List<float2> uvs = [];
    private readonly List<FaceGroup> groups = [];

    private ObjMesh()
    {
    }

    public int PositionCount => positions.Count;
    public int TriangleCount => groups.Sum(static group => group.PositionIndices.Count / 3);

    public float3 Center
    {
        get
        {
            var min = new float3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new float3(float.MinValue, float.MinValue, float.MinValue);
            foreach (var position in positions)
            {
                min = new float3(Math.Min(min.X, position.X), Math.Min(min.Y, position.Y), Math.Min(min.Z, position.Z));
                max = new float3(Math.Max(max.X, position.X), Math.Max(max.Y, position.Y), Math.Max(max.Z, position.Z));
            }

            return (min + max) * 0.5f;
        }
    }

    public float Radius
    {
        get
        {
            var center = Center;
            var radius = 0.0f;
            foreach (var position in positions)
            {
                radius = Math.Max(radius, CpuMath.Length(position - center));
            }

            return radius;
        }
    }

    public static ObjMesh Load(string objPath, string baseDirectory, MaterialLibrary materials)
    {
        var mesh = new ObjMesh();
        using var reader = File.OpenText(objPath);
        var currentGroup = new FaceGroup("default");

        while (reader.ReadLine() is { } line)
        {
            var parts = SplitTokens(line);
            if (parts.Length == 0 || parts[0].StartsWith('#'))
            {
                continue;
            }

            switch (parts[0])
            {
                case "mtllib" when parts.Length >= 2:
                    materials.Load(Path.Combine(baseDirectory, parts[1]), baseDirectory);
                    break;
                case "v" when parts.Length >= 4:
                    mesh.positions.Add(new float3(ParseFloat(parts[1]), ParseFloat(parts[2]), ParseFloat(parts[3])));
                    break;
                case "vt" when parts.Length >= 3:
                    mesh.uvs.Add(new float2(ParseFloat(parts[1]), ParseFloat(parts[2])));
                    break;
                case "vn" when parts.Length >= 4:
                    mesh.normals.Add(new float3(ParseFloat(parts[1]), ParseFloat(parts[2]), ParseFloat(parts[3])));
                    break;
                case "usemtl" when parts.Length >= 2:
                    mesh.CommitGroup(currentGroup);
                    currentGroup = new FaceGroup(parts[1]);
                    break;
                case "f" when parts.Length >= 4:
                    mesh.AddFace(currentGroup, parts.AsSpan(1));
                    break;
            }
        }

        mesh.CommitGroup(currentGroup);
        if (mesh.groups.Count == 0)
        {
            throw new InvalidDataException("OBJ file does not contain any renderable faces.");
        }

        return mesh;
    }

    public SponzaVertex[] Flatten(MaterialLibrary materials, TextureAtlas atlas)
    {
        var result = new List<SponzaVertex>(TriangleCount * 3);
        foreach (var group in groups)
        {
            var material = materials.Find(group.MaterialName);
            var atlasTransform = atlas.GetUvTransform(material?.AtlasSlot ?? 0);
            for (var i = 0; i < group.PositionIndices.Count; i++)
            {
                var position = positions[group.PositionIndices[i]];
                var normal = group.NormalIndices[i] >= 0 && group.NormalIndices[i] < normals.Count
                    ? normals[group.NormalIndices[i]]
                    : new float3(0.0f, 1.0f, 0.0f);
                var uv = group.UvIndices[i] >= 0 && group.UvIndices[i] < uvs.Count
                    ? uvs[group.UvIndices[i]]
                    : float2.Zero;
                result.Add(new SponzaVertex
                {
                    Position = position, Normal = normal, Uv = uv, AtlasTransform = atlasTransform
                });
            }
        }

        return [.. result];
    }

    private void AddFace(FaceGroup group, ReadOnlySpan<string> faceVertices)
    {
        Span<VertexIndex> polygon = stackalloc VertexIndex[faceVertices.Length];
        for (var i = 0; i < faceVertices.Length; i++)
        {
            polygon[i] = ParseVertex(faceVertices[i]);
        }

        for (var i = 1; i < polygon.Length - 1; i++)
        {
            AddTriangle(group, polygon[0], polygon[i], polygon[i + 1]);
        }
    }

    private void AddTriangle(FaceGroup group, VertexIndex a, VertexIndex b, VertexIndex c)
    {
        AddVertex(group, a);
        AddVertex(group, b);
        AddVertex(group, c);
    }

    private static void AddVertex(FaceGroup group, VertexIndex vertex)
    {
        group.PositionIndices.Add(vertex.Position);
        group.UvIndices.Add(vertex.Uv);
        group.NormalIndices.Add(vertex.Normal);
    }

    private VertexIndex ParseVertex(string value)
    {
        var fields = value.Split('/');
        var position = ResolveIndex(int.Parse(fields[0], CultureInfo.InvariantCulture), positions.Count);
        var uv = fields.Length > 1 && fields[1].Length > 0
            ? ResolveIndex(int.Parse(fields[1], CultureInfo.InvariantCulture), uvs.Count)
            : -1;
        var normal = fields.Length > 2 && fields[2].Length > 0
            ? ResolveIndex(int.Parse(fields[2], CultureInfo.InvariantCulture), normals.Count)
            : -1;
        return new VertexIndex(position, uv, normal);
    }

    private void CommitGroup(FaceGroup group)
    {
        if (group.PositionIndices.Count > 0)
        {
            groups.Add(group);
        }
    }

    private static int ResolveIndex(int index, int count)
        => index > 0 ? index - 1 : count + index;

    private static string[] SplitTokens(string line)
        => line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    private static float ParseFloat(string value)
        => float.Parse(value, CultureInfo.InvariantCulture);

    private sealed class FaceGroup(string materialName)
    {
        public string MaterialName { get; } = materialName;
        public List<int> PositionIndices { get; } = [];
        public List<int> UvIndices { get; } = [];
        public List<int> NormalIndices { get; } = [];
    }

    private readonly record struct VertexIndex(int Position, int Uv, int Normal);
}

file struct Camera(float3 position, float yaw, float pitch)
{
    public float3 Position = position;
    public float Yaw = yaw;
    public float Pitch = pitch;

    public float3 FlatForward
    {
        get
        {
            var sinYaw = MathF.Sin(Yaw);
            var cosYaw = MathF.Cos(Yaw);
            return CpuMath.Normalize(new float3(sinYaw, 0.0f, -cosYaw));
        }
    }

    public float3 Right
    {
        get
        {
            var cosYaw = MathF.Cos(Yaw);
            var sinYaw = MathF.Sin(Yaw);
            return CpuMath.Normalize(new float3(cosYaw, 0.0f, sinYaw));
        }
    }
}

file static class MatrixMath
{
    public static float4x4 Scale(float scale)
        => new(
            new float4(scale, 0.0f, 0.0f, 0.0f),
            new float4(0.0f, scale, 0.0f, 0.0f),
            new float4(0.0f, 0.0f, scale, 0.0f),
            new float4(0.0f, 0.0f, 0.0f, 1.0f));

    public static float4x4 Translation(float3 offset)
        => new(
            new float4(1.0f, 0.0f, 0.0f, 0.0f),
            new float4(0.0f, 1.0f, 0.0f, 0.0f),
            new float4(0.0f, 0.0f, 1.0f, 0.0f),
            new float4(offset, 1.0f));

    public static float4x4 PerspectiveVk(float fovDegrees, float aspect, float near, float far)
    {
        var tanHalfFov = MathF.Tan(fovDegrees * 0.5f * MathF.PI / 180.0f);
        return new float4x4(
            new float4(1.0f / (aspect * tanHalfFov), 0.0f, 0.0f, 0.0f),
            new float4(0.0f, -1.0f / tanHalfFov, 0.0f, 0.0f),
            new float4(0.0f, 0.0f, far / (near - far), -1.0f),
            new float4(0.0f, 0.0f, (near * far) / (near - far), 0.0f));
    }

    public static float4x4 CameraView(float3 position, float yaw, float pitch)
    {
        var cosYaw = MathF.Cos(yaw);
        var sinYaw = MathF.Sin(yaw);
        var cosPitch = MathF.Cos(pitch);
        var sinPitch = MathF.Sin(pitch);
        var forward = CpuMath.Normalize(new float3(sinYaw * cosPitch, sinPitch, -cosYaw * cosPitch));
        var right = CpuMath.Normalize(CpuMath.Cross(forward, new float3(0.0f, 1.0f, 0.0f)));
        var up = CpuMath.Cross(right, forward);

        return new float4x4(
            new float4(right.X, up.X, -forward.X, 0.0f),
            new float4(right.Y, up.Y, -forward.Y, 0.0f),
            new float4(right.Z, up.Z, -forward.Z, 0.0f),
            new float4(-CpuMath.Dot(right, position), -CpuMath.Dot(up, position), CpuMath.Dot(forward, position),
                1.0f));
    }
}

file static class CpuMath
{
    public static float Dot(float3 a, float3 b)
        => (a.X * b.X) + (a.Y * b.Y) + (a.Z * b.Z);

    public static float3 Cross(float3 a, float3 b)
        => new(
            (a.Y * b.Z) - (a.Z * b.Y),
            (a.Z * b.X) - (a.X * b.Z),
            (a.X * b.Y) - (a.Y * b.X));

    public static float Length(float3 value)
        => MathF.Sqrt(Dot(value, value));

    public static float3 Normalize(float3 value)
    {
        var length = Length(value);
        return length == 0.0f ? float3.Zero : value / length;
    }
}

file sealed class TgaImage
{
    private TgaImage(int width, int height, Rgba32[] pixels)
    {
        Width = width;
        Height = height;
        Pixels = pixels;
    }

    public int Width { get; }
    public int Height { get; }
    public Rgba32[] Pixels { get; }

    public static TgaImage Load(string path)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 18)
        {
            throw new InvalidDataException("TGA file is shorter than the header.");
        }

        var idLength = bytes[0];
        var colorMapType = bytes[1];
        var imageType = bytes[2];
        if (colorMapType != 0 || imageType is not (2 or 3))
        {
            throw new NotSupportedException("Only uncompressed grayscale and true-color TGA files are supported.");
        }

        var width = bytes[12] | (bytes[13] << 8);
        var height = bytes[14] | (bytes[15] << 8);
        var bitsPerPixel = bytes[16];
        var descriptor = bytes[17];
        var pixelSize = bitsPerPixel / 8;
        if (width <= 0 || height <= 0 || bitsPerPixel is not (8 or 24 or 32))
        {
            throw new NotSupportedException($"Unsupported TGA format {width}x{height}x{bitsPerPixel}.");
        }

        var dataOffset = 18 + idLength;
        var required = checked(width * height * pixelSize);
        if (bytes.Length - dataOffset < required)
        {
            throw new InvalidDataException("TGA file does not contain enough pixel data.");
        }

        var pixels = new Rgba32[checked(width * height)];
        var topOrigin = (descriptor & 0x20) != 0;
        for (var y = 0; y < height; y++)
        {
            var srcY = topOrigin ? y : height - 1 - y;
            for (var x = 0; x < width; x++)
            {
                var src = dataOffset + (((srcY * width) + x) * pixelSize);
                pixels[(y * width) + x] = imageType == 3
                    ? new Rgba32(bytes[src], bytes[src], bytes[src], 255)
                    : new Rgba32(bytes[src + 2], bytes[src + 1], bytes[src],
                        pixelSize == 4 ? bytes[src + 3] : (byte)255);
            }
        }

        return new TgaImage(width, height, pixels);
    }
}
