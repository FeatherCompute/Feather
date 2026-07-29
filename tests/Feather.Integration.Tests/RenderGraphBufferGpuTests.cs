using Feather.Interop;
using Feather.Math;
using Feather.RenderGraph;
using Feather.Resources;

namespace Feather.Integration.Tests;

public sealed class RenderGraphBufferGpuTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeKernelPublishesTypedGpuBufferThroughRenderContext()
    {
        var directions = new[]
        {
            new float3(1, 2, 3),
            new float3(4, 5, 6),
            new float3(-1, 2, 4)
        };
        var weights = new[] { 1, 2, 3 };
        using var directionBuffer = GPU.CreateBuffer(directions, BufferAccess.ReadOnly);
        using var weightBuffer = GPU.CreateBuffer(weights, BufferAccess.ReadOnly);
        using var outputBuffer = GPU.CreateBuffer<float>(directions.Length, BufferAccess.ReadWrite);

        var path = GPU.DispatchAndGetPath(
            new WeightedDirectionKernel(
                directionBuffer.AsReadOnly(),
                weightBuffer.AsReadOnly(),
                outputBuffer.AsReadWrite()),
            directions.Length);
        var backend = new CapturingRenderContextBackend();
        var context = new RenderContext(backend);
        var outputHandle = new BufferHandle<float>(42);

        context.SetBufferOutput(outputHandle, outputBuffer, path);

        Assert.Equal(DispatchPath.TypedEasyGpu, path);
        Assert.Equal(outputHandle.Value, backend.OutputHandle);
        Assert.Equal(path, backend.DispatchPath);
        Assert.Equal(new float[] { 6, 30, 15 }, Assert.IsType<float[]>(backend.Values));
        Assert.Equal(1, backend.ReadbackReportCount);
        Assert.True(backend.ReadbackDuration >= TimeSpan.Zero);
    }

    private sealed class CapturingRenderContextBackend : IRenderContextBackend
    {
        public int Width => 1;
        public int Height => 1;
        public SampleCount SampleCount => SampleCount.X1;
        public ulong OutputHandle { get; private set; }
        public Array? Values { get; private set; }
        public DispatchPath DispatchPath { get; private set; }
        public int ReadbackReportCount { get; private set; }
        public TimeSpan ReadbackDuration { get; private set; }

        public SceneGeometry GetSceneGeometry(SceneGeometryHandle handle)
            => new(Array.Empty<SceneVertex>(), Array.Empty<uint>());

        public RenderCamera GetCamera(CameraHandle handle)
            => new(float4x4.Identity);

        public ReadOnlyMemory<Rgba8> GetColorInput(TextureHandle handle)
            => ReadOnlyMemory<Rgba8>.Empty;

        public void SetColorOutput(
            TextureHandle handle,
            Rgba8[] pixels,
            DispatchPath dispatchPath)
        {
        }

        public void SetBufferOutput<T>(
            BufferHandle<T> handle,
            T[] values,
            DispatchPath dispatchPath)
            where T : unmanaged
        {
            OutputHandle = handle.Value;
            Values = values;
            DispatchPath = dispatchPath;
        }

        public void ReportGpuReadback(TimeSpan elapsed)
        {
            ReadbackReportCount++;
            ReadbackDuration += elapsed;
        }
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct WeightedDirectionKernel(
    ReadOnlyBuffer<float3> directions,
    ReadOnlyBuffer<int> weights,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int index = ThreadIds.X;
        float3 direction = directions[index];
        float sum = 0.0f;
        for (int component = 0; component < 3; component++)
        {
            if (component == 0)
            {
                sum += direction.X;
            }
            else if (component == 1)
            {
                sum += direction.Y;
            }
            else
            {
                sum += direction.Z;
            }
        }

        int weight = weights[index];
        if (weight > 1)
        {
            sum *= weight;
        }
        output[index] = sum;
    }
}
