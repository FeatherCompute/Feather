using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

public sealed class RenderSchedulingTests
{
    private const string SceneFingerprint = "scene-a";

    [Fact]
    public void ProgressiveViewPublishesFirstIntervalAndFinalIterations()
    {
        var scheduler = new RenderScheduler();
        var graph = Graph(RenderExecutionMode.Progressive, targetSamples: 4, previewEverySamples: 2);
        var state = scheduler.Prepare(Request(1), graph, SceneFingerprint);

        var first = state.CompleteIteration(graph);
        var second = state.CompleteIteration(graph);
        var third = state.CompleteIteration(graph);
        var fourth = state.CompleteIteration(graph);

        Assert.True(first.PublishFrame);
        Assert.True(first.HistoryReset);
        Assert.True(first.NeedsMoreWork);
        Assert.True(second.PublishFrame);
        Assert.False(third.PublishFrame);
        Assert.True(fourth.PublishFrame);
        Assert.True(fourth.Completed);
        Assert.False(fourth.NeedsMoreWork);
        Assert.Equal(4, fourth.AccumulatedSamples);
        Assert.Equal(1, fourth.ResetCount);
    }

    [Fact]
    public void RealtimeStatePersistsUntilARelevantInputChanges()
    {
        var scheduler = new RenderScheduler();
        var graph = Graph(RenderExecutionMode.Realtime);
        var request = Request(1);
        var state = scheduler.Prepare(request, graph, SceneFingerprint);
        state.CommitHistory(new Dictionary<string, GraphHistoryEntry>
        {
            ["taa"] = GraphHistoryEntry.FromFrame(Frame(2, 2, 7))
        });
        _ = state.CompleteIteration(graph);

        var unchanged = scheduler.Prepare(Request(999), graph, SceneFingerprint);
        Assert.Same(state, unchanged);
        Assert.Single(unchanged.History);
        Assert.False(unchanged.ResetOccurred);

        var resized = scheduler.Prepare(Request(1000, width: 3), graph, SceneFingerprint);
        Assert.Same(state, resized);
        Assert.Empty(resized.History);
        Assert.True(resized.ResetOccurred);
        Assert.Equal(2, resized.ResetCount);
        Assert.Equal(0, resized.AccumulatedSamples);
    }

    [Fact]
    public void OnDemandRequestIdStartsANewEpoch()
    {
        var scheduler = new RenderScheduler();
        var graph = Graph(RenderExecutionMode.OnDemand);
        var state = scheduler.Prepare(Request(10), graph, SceneFingerprint);
        _ = state.CompleteIteration(graph);

        scheduler.Prepare(Request(11), graph, SceneFingerprint);

        Assert.True(state.ResetOccurred);
        Assert.Equal(2, state.ResetCount);
        Assert.Equal(0, state.Iteration);
    }

    [Fact]
    public void PassReloadClearsTemporalHistory()
    {
        var scheduler = new RenderScheduler();
        var graph = Graph(RenderExecutionMode.Progressive, targetSamples: 0);
        var state = scheduler.Prepare(Request(1), graph, SceneFingerprint);
        state.CommitHistory(new Dictionary<string, GraphHistoryEntry>
        {
            ["accumulation"] = GraphHistoryEntry.FromFrame(Frame(1, 1, 42))
        });
        _ = state.CompleteIteration(graph);

        state.ResetForPassReload(graph);

        Assert.Empty(state.History);
        Assert.Equal(0, state.Iteration);
        Assert.Equal(0, state.AccumulatedSamples);
        Assert.Equal(2, state.ResetCount);
        Assert.True(state.ResetOccurred);
    }

    [Fact]
    public void PassBuildChangesResetEveryViewWhenItNextRenders()
    {
        var scheduler = new RenderScheduler();
        var graph1 = Graph(RenderExecutionMode.Progressive, targetSamples: 0);
        var graph2 = Graph(RenderExecutionMode.Progressive, targetSamples: 0, viewId: "view-2");
        var view1 = scheduler.Prepare(Request(1), graph1, SceneFingerprint);
        var view2 = scheduler.Prepare(Request(1, viewId: "view-2"), graph2, SceneFingerprint);
        view1.PreparePassBuild("build-a", graph1);
        view2.PreparePassBuild("build-a", graph2);
        _ = view1.CompleteIteration(graph1);
        _ = view2.CompleteIteration(graph2);

        view1.PreparePassBuild("build-b", graph1);
        view2.PreparePassBuild("build-b", graph2);

        Assert.Equal(0, view1.Iteration);
        Assert.Equal(0, view2.Iteration);
        Assert.Equal(2, view1.ResetCount);
        Assert.Equal(2, view2.ResetCount);
    }

    private static ResolvedRenderRequest Request(
        ulong requestId,
        int width = 2,
        string viewId = "view-1")
        => new(
            requestId,
            ProtocolFixture.GenerationId,
            viewId,
            "scene",
            "graph",
            "manifest",
            "output",
            width,
            2,
            float4x4.Identity,
            float4x4.Identity,
            new float3(0.0f, 0.0f, 0.0f),
            RenderPurpose.Interactive);

    private static RenderGraphExecution Graph(
        RenderExecutionMode mode,
        int targetSamples = 1,
        int previewEverySamples = 1,
        string viewId = "view-1")
    {
        var scene = new GraphNode { NodeId = "scene", Kind = "scene" };
        var pass = new GraphNode
        {
            NodeId = "pass",
            Kind = "pass",
            PassGuid = RenderGraphDocument.MinimalRasterPassGuid,
            TypeName = RenderGraphDocument.MinimalRasterPassType
        };
        var output = new GraphNode { NodeId = "output", Kind = "output" };
        var outputLink = new GraphLink
        {
            FromNode = pass.NodeId,
            FromSocket = RenderGraphDocument.ColorOutputSocketGuid,
            ToNode = output.NodeId,
            ToSocket = RenderGraphDocument.OutputColorSocketGuid
        };
        return new RenderGraphExecution(
            ProtocolFixture.GenerationId,
            "graph-1",
            viewId,
            "CUSTOM",
            mode,
            SampleCount.X1,
            targetSamples,
            1,
            previewEverySamples,
            "Combined",
            [scene, pass, output],
            [outputLink],
            [pass],
            scene,
            output,
            outputLink,
            [],
            []);
    }

    private static RenderedFrame Frame(int width, int height, byte value)
    {
        var pixels = new Rgba8[checked(width * height)];
        Array.Fill(pixels, new Rgba8(value, value, value, 255));
        return new RenderedFrame(width, height, pixels, DispatchPath.None);
    }
}
