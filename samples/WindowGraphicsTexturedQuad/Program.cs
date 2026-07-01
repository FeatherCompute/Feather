using Feather;
using Feather.Graphics;
using Feather.Math;
using Feather.Resources;
using Feather.Windowing;

using var window = GpuWindow.Create(new()
{
    Width = 800,
    Height = 600,
    Title = "Feather Textured Quad"
});
using var presenter = window.CreateTexturePresenter();
using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(window.Width, window.Height, PixelFormat.Rgba8);
using var vertices = GPU.CreateBuffer<float4>(
[
    new float4(-0.85f, -0.75f, 0, 1),
    new float4(0.85f, -0.75f, 0, 1),
    new float4(0.85f, 0.75f, 0, 1),
    new float4(-0.85f, 0.75f, 0, 1)
]);
using var indices = GPU.CreateIndexBuffer<uint>([0, 1, 2, 0, 2, 3]);
using var texture = GPU.CreateTexture2D<Rgba32, float4>(2, 2, PixelFormat.Rgba8, TextureAccess.Sampled);
using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
using var pipeline = GPU.CreateGraphicsPipeline<QuadVS, QuadFS, float4>();

texture.Upload(
[
    new Rgba32(255, 70, 80, 255),
    new Rgba32(50, 210, 120, 255),
    new Rgba32(70, 130, 255, 255),
    new Rgba32(255, 220, 70, 255)
]);

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

    pipeline.DrawIndexed(
        new QuadVS(vertices.AsReadOnly()),
        new QuadFS(texture.AsSampled(), sampler),
        color,
        indices);
    presenter.Present(color);
}

public readonly record struct Rgba32(byte R, byte G, byte B, byte A);

[VertexShader]
public readonly partial struct QuadVS(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    public float4 Execute()
    {
        return vertices[VertexIds.Index];
    }
}

[FragmentShader]
public readonly partial struct QuadFS(SampledTexture2D<float4> texture, SamplerState sampler) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return texture.Sample(sampler, input.XY);
    }
}
