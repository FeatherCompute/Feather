using Feather;
using Feather.Math;
using Feather.Resources;
using Feather.Windowing;

using var window = GpuWindow.Create(new()
{
    Width = 800,
    Height = 450,
    Title = "Feather Compute Texture"
});
using var presenter = window.CreateTexturePresenter();
using var color = GPU.CreateRenderTexture2D<float4, float4>(window.Width, window.Height, PixelFormat.Rgba32Float);

var frame = 0;
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

    GPU.Dispatch(new ComputePixels(color.AsReadWrite(), new Uniform<int>(frame)), color.Size);
    presenter.Present(color);
    frame++;
}

[Kernel]
[ThreadGroupSize(8, 8, 1)]
public readonly partial struct ComputePixels(ReadWriteTexture2D<float4> output, Uniform<int> frame) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        int t = frame.Value;
        float r = ((p.X + t) & 255) / 255.0f;
        float g = ((p.Y * 2 + t) & 255) / 255.0f;
        float b = ((p.X + p.Y + t * 3) & 255) / 255.0f;
        output[p] = new float4(r, g, b, 1.0f);
    }
}
