using Feather.RenderGraph;

namespace Feather.Blender.RenderHost;

internal sealed class RenderHostRunner : IDisposable
{
    private readonly MinimalRasterRenderer renderer = new();
    private readonly ProjectPassAssemblyManager projectPasses = new();
    private readonly RenderScheduler scheduler = new();
    private readonly RenderSceneCache sceneCache = new();

    internal WeakReference? LastUnloadedPassContextForTesting
        => projectPasses.LastUnloadedContextForTesting;

    public RenderHostResult RenderOnce(string requestPath)
    {
        var started = System.Diagnostics.Stopwatch.StartNew();
        var stage = System.Diagnostics.Stopwatch.StartNew();
        var request = RenderRequest.Load(requestPath);
        var graph = RenderGraphDocument.Load(request.GraphPath);
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
        stage.Stop();
        var protocolLoadMilliseconds = stage.Elapsed.TotalMilliseconds;

        var cachedScene = sceneCache.Resolve(request.ScenePath);
        if (!string.Equals(request.GenerationId, cachedScene.GenerationId, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Render request generation '{request.GenerationId}' does not match scene generation '{cachedScene.GenerationId}'.");
        }
        var sceneLoadMilliseconds = cachedScene.LoadMilliseconds;
        var viewState = scheduler.Prepare(request, graph, cachedScene.ContentFingerprint);
        var scene = cachedScene.Resources;
        var sceneBuildMilliseconds = cachedScene.BuildMilliseconds;
        var geometry = scene.Geometry;
        RenderedFrame frame;
        string buildId;
        string passType;
        var passCount = 1;
        var passReloaded = false;
        var gpuReadbackMilliseconds = 0.0;
        IReadOnlyList<PassExecutionTiming> passTimings = Array.Empty<PassExecutionTiming>();
        stage.Restart();
        if (request.ManifestPath is not null)
        {
            var execution = projectPasses.Execute(
                request.ManifestPath,
                graph,
                scene,
                request.Width,
                request.Height,
                request.ViewProjection,
                request.InverseViewProjection,
                request.CameraPosition,
                request.Purpose,
                cachedScene.ContentFingerprint,
                viewState);
            frame = execution.Frame;
            buildId = execution.BuildId;
            passType = execution.PassType;
            passCount = execution.PassCount;
            passReloaded = execution.Reloaded;
            gpuReadbackMilliseconds = execution.GpuReadbackMilliseconds;
            passTimings = execution.PassTimings;
        }
        else
        {
            graph.RequireLegacyMinimalRaster();
            frame = renderer.Render(
                geometry,
                request.Width,
                request.Height,
                request.ViewProjection,
                graph.SampleCount,
                graph.Settings);
            buildId = "builtin";
            passType = RenderGraphDocument.MinimalRasterPassType;
        }
        stage.Stop();
        var passExecutionMilliseconds = stage.Elapsed.TotalMilliseconds;
        var iteration = viewState.CompleteIteration(graph);
        var frameWriteMilliseconds = 0.0;
        if (iteration.PublishFrame)
        {
            stage.Restart();
            FrameFileWriter.WriteAtomic(request.OutputPath, request.RequestId, frame);
            stage.Stop();
            frameWriteMilliseconds = stage.Elapsed.TotalMilliseconds;
        }
        started.Stop();

        return new RenderHostResult(
            request.RequestId,
            request.OutputPath,
            request.Width,
            request.Height,
            geometry.Vertices.Length,
            geometry.Indices.Length / 3,
            frame.DispatchPath.ToString(),
            buildId,
            passType,
            passCount,
            passReloaded,
            graph.ExecutionModeName,
            request.Purpose == RenderPurpose.Final ? "FINAL" : "INTERACTIVE",
            graph.PublishedAov,
            graph.TargetSamples,
            iteration.Iteration,
            iteration.AccumulatedSamples,
            iteration.PublishFrame,
            iteration.Completed,
            iteration.NeedsMoreWork,
            iteration.HistoryReset,
            iteration.ResetCount,
            protocolLoadMilliseconds,
            sceneLoadMilliseconds,
            sceneBuildMilliseconds,
            passExecutionMilliseconds,
            gpuReadbackMilliseconds,
            frameWriteMilliseconds,
            started.Elapsed.TotalMilliseconds,
            passTimings,
            scene.Diagnostics);
    }

    public void Dispose()
    {
        projectPasses.Dispose();
        renderer.Dispose();
    }
}

internal sealed record RenderHostResult(
    ulong RequestId,
    string OutputPath,
    int Width,
    int Height,
    int VertexCount,
    int TriangleCount,
    string DispatchPath,
    string BuildId,
    string PassType,
    int PassCount,
    bool PassReloaded,
    string ExecutionMode,
    string Purpose,
    string Aov,
    int TargetSamples,
    long Iteration,
    long AccumulatedSamples,
    bool FramePublished,
    bool Completed,
    bool NeedsMoreWork,
    bool HistoryReset,
    long ResetCount,
    double ProtocolLoadMilliseconds,
    double SceneLoadMilliseconds,
    double SceneBuildMilliseconds,
    double PassExecutionMilliseconds,
    double GpuReadbackMilliseconds,
    double FrameWriteMilliseconds,
    double TotalMilliseconds,
    IReadOnlyList<PassExecutionTiming> PassTimings,
    IReadOnlyList<RenderHostDiagnostic> Diagnostics);
