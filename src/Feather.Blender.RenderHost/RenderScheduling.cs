using Feather.Math;

namespace Feather.Blender.RenderHost;

internal sealed class RenderScheduler
{
    private readonly Dictionary<string, RenderViewState> views = new(StringComparer.Ordinal);

    public RenderViewState Prepare(
        ResolvedRenderRequest request,
        RenderGraphExecution graph,
        string sceneFingerprint)
    {
        if (!views.TryGetValue(graph.ViewId, out var state))
        {
            state = new RenderViewState(graph.ViewId);
            views.Add(graph.ViewId, state);
        }

        var requestEpoch = graph.ExecutionMode is RenderExecutionMode.OnDemand or RenderExecutionMode.Offline
            ? request.RequestId
            : 0UL;
        state.Prepare(
            new RenderInputIdentity(
                sceneFingerprint,
                graph.GraphId,
                graph.GraphFingerprint,
                graph.ViewId,
                request.Width,
                request.Height,
                request.ViewProjection,
                request.InverseViewProjection,
                request.CameraPosition,
                graph.ExecutionMode,
                graph.TargetSamples,
                graph.SamplesPerIteration,
                graph.PreviewEverySamples,
                graph.OutputLink.FromNode,
                graph.OutputLink.FromSocket,
                graph.SelectedAov,
                requestEpoch),
            graph);
        return state;
    }
}

internal sealed class RenderViewState
{
    private readonly Dictionary<string, GraphHistoryEntry> history = new(StringComparer.Ordinal);
    private RenderInputIdentity? identity;
    private string? passBuildId;
    private long nextPreviewSample;

    public RenderViewState(string viewId)
    {
        ViewId = viewId;
    }

    public string ViewId { get; }
    public long Iteration { get; private set; }
    public long AccumulatedSamples { get; private set; }
    public long ResetCount { get; private set; }
    public bool ResetOccurred { get; private set; }
    public IReadOnlyDictionary<string, GraphHistoryEntry> History => history;

    public void Prepare(RenderInputIdentity nextIdentity, RenderGraphExecution graph)
    {
        ResetOccurred = false;
        if (identity != nextIdentity)
        {
            identity = nextIdentity;
            Reset(graph);
        }
    }

    public void ResetForPassReload(RenderGraphExecution graph)
    {
        if (Iteration == 0 && AccumulatedSamples == 0 && history.Count == 0)
        {
            ResetOccurred = true;
            return;
        }
        Reset(graph);
    }

    public void PreparePassBuild(string buildId, RenderGraphExecution graph)
    {
        if (string.Equals(passBuildId, buildId, StringComparison.Ordinal))
        {
            return;
        }
        passBuildId = buildId;
        ResetForPassReload(graph);
    }

    public void CommitHistory(IReadOnlyDictionary<string, GraphHistoryEntry> updates)
    {
        foreach (var (key, entry) in updates)
        {
            history[key] = entry;
        }
    }

    public RenderIterationStatus CompleteIteration(RenderGraphExecution graph)
    {
        var firstIteration = Iteration == 0;
        Iteration++;
        AccumulatedSamples = checked(AccumulatedSamples + graph.SamplesPerIteration);

        var needsMoreWork = graph.ExecutionMode switch
        {
            RenderExecutionMode.Progressive =>
                graph.TargetSamples == 0 || AccumulatedSamples < graph.TargetSamples,
            RenderExecutionMode.Offline => AccumulatedSamples < graph.TargetSamples,
            _ => false
        };
        var completed = !needsMoreWork &&
            (graph.ExecutionMode != RenderExecutionMode.Progressive || graph.TargetSamples > 0);
        var publish = firstIteration || completed || AccumulatedSamples >= nextPreviewSample;
        if (AccumulatedSamples >= nextPreviewSample)
        {
            do
            {
                nextPreviewSample = checked(nextPreviewSample + graph.PreviewEverySamples);
            }
            while (nextPreviewSample <= AccumulatedSamples);
        }

        return new RenderIterationStatus(
            Iteration,
            AccumulatedSamples,
            publish,
            completed,
            needsMoreWork,
            ResetOccurred,
            ResetCount);
    }

    private void Reset(RenderGraphExecution graph)
    {
        history.Clear();
        Iteration = 0;
        AccumulatedSamples = 0;
        nextPreviewSample = graph.PreviewEverySamples;
        ResetCount++;
        ResetOccurred = true;
    }
}

internal readonly record struct RenderIterationStatus(
    long Iteration,
    long AccumulatedSamples,
    bool PublishFrame,
    bool Completed,
    bool NeedsMoreWork,
    bool HistoryReset,
    long ResetCount);

internal readonly record struct RenderInputIdentity(
    string SceneFingerprint,
    string GraphId,
    string GraphFingerprint,
    string ViewId,
    int Width,
    int Height,
    float4x4 ViewProjection,
    float4x4 InverseViewProjection,
    float3 CameraPosition,
    RenderExecutionMode ExecutionMode,
    int TargetSamples,
    int SamplesPerIteration,
    int PreviewEverySamples,
    string OutputNodeId,
    string OutputSocketGuid,
    string SelectedAov,
    ulong RequestEpoch);
