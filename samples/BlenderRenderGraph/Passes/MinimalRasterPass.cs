using Feather.RenderGraph;

namespace BlenderRenderGraph.Passes;

[FeatherPass(
    "01c671a1-9b4e-5cab-b7e1-c101348af596",
    Name = "Minimal Raster",
    Category = "Raster",
    Version = 1)]
public sealed class MinimalRasterPass : IRasterPass
{
    [Input("6d6eb2d5-bb7a-55a4-a85a-c58e36715c53")]
    public SceneGeometryHandle Geometry { get; init; }

    [Input("a6eed590-b632-5f91-a69d-09b6eb4bb5ac")]
    public MaterialTableHandle Materials { get; init; }

    [Input("cc78191c-ac9a-57b6-bcac-91cce5e298f5")]
    public CameraHandle Camera { get; init; }

    [Output(
        "bd711ea6-36f9-56cd-863a-cfec58727a46",
        Format = TextureFormat.Rgba16Float)]
    public TextureHandle Color { get; init; }

    [Parameter(
        "2a1df649-2b96-558f-ae30-9d9bf6858d43",
        Min = 0.0,
        Max = 8.0)]
    public float Exposure { get; set; } = 1.0f;

    public void Execute(RenderContext context)
    {
        // The Blender RenderHost supplies resources and records commands through this context.
    }
}
