using Feather;
using Feather.Graphics;
using Feather.Math;
using Feather.Resources;
using Feather.Windowing;

using var window = GpuWindow.Create(new()
{
    Width = 800,
    Height = 600,
    Title = "Feather Graphics Triangle"
});
using var presenter = window.CreateTexturePresenter();
using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(window.Width, window.Height, PixelFormat.Rgba8);
using var vertices = GPU.CreateBuffer<float4>(
[
    new float4(-0.8f, -0.7f, 0, 1),
    new float4(0.8f, -0.7f, 0, 1),
    new float4(0, 0.75f, 0, 1)
]);
using var pipeline = GPU.CreateGraphicsPipeline<TriangleVS, TriangleFS, float4>();

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

    pipeline.Draw(new TriangleVS(vertices.AsReadOnly()), new TriangleFS(), color, vertexCount: 3);
    presenter.Present(color);
}

public readonly record struct Rgba32(byte R, byte G, byte B, byte A);

[VertexShader]
public readonly partial struct TriangleVS(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    public float4 Execute()
    {
        return vertices[VertexIds.Index];
    }
}

[FragmentShader]
public readonly partial struct TriangleFS : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return input;
    }
}
