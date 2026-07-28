namespace Feather.Blender.RenderHost;

internal sealed class RenderHostRunner : IDisposable
{
    private readonly MinimalRasterRenderer renderer = new();

    public RenderHostResult RenderOnce(string requestPath)
    {
        var started = System.Diagnostics.Stopwatch.StartNew();
        var request = RenderRequest.Load(requestPath);
        var graph = RenderGraphDocument.LoadMinimalRaster(request.GraphPath);
        if (!string.Equals(request.GenerationId, graph.GenerationId, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Render request generation '{request.GenerationId}' does not match graph generation '{graph.GenerationId}'.");
        }
        if (!string.Equals(request.ViewId, graph.ViewId, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Render request view '{request.ViewId}' does not match graph view '{graph.ViewId}'.");
        }

        var snapshot = SceneSnapshot.Load(request.ScenePath);
        if (!string.Equals(request.GenerationId, snapshot.Metadata.GenerationId, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Render request generation '{request.GenerationId}' does not match scene generation '{snapshot.Metadata.GenerationId}'.");
        }
        var geometry = SceneGeometryBuilder.Build(snapshot);
        var frame = renderer.Render(
            geometry,
            request.Width,
            request.Height,
            request.ViewProjection,
            graph.SampleCount,
            graph.Settings);
        FrameFileWriter.WriteAtomic(request.OutputPath, request.RequestId, frame);
        started.Stop();

        return new RenderHostResult(
            request.RequestId,
            request.OutputPath,
            request.Width,
            request.Height,
            geometry.Vertices.Length,
            geometry.Indices.Length / 3,
            frame.DispatchPath.ToString(),
            started.Elapsed.TotalMilliseconds);
    }

    public void Dispose() => renderer.Dispose();
}

internal sealed record RenderHostResult(
    ulong RequestId,
    string OutputPath,
    int Width,
    int Height,
    int VertexCount,
    int TriangleCount,
    string DispatchPath,
    double TotalMilliseconds);
