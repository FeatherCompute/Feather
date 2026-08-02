using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

public sealed class TemporalRenderHostIntegrationTests
{
    private const string AccumulatorPassGuid = "a805ce16-ed91-4e72-89ba-43de0ed4f977";
    private const string AccumulatorInputGuid = "18389027-71c7-4ffc-bc22-811f14ab3fc4";
    private const string AccumulatorOutputGuid = "7a21760f-fdfd-4419-a933-32e8fdb23665";
    private const string IncrementParameterGuid = "f4ff766f-e28c-4ddb-ac1e-0a81b28cabfa";
    private const string FailurePassGuid = "0b0a3fc1-665b-47d6-98f0-1a91d63f555d";
    private const string FailureInputGuid = "03f03b52-6ff0-4e9e-a58e-3caed1af8015";
    private const string FailureOutputGuid = "0aa66062-cabc-432c-980a-e3db67a4b7dc";
    private const string FailurePathParameterGuid = "2885a6f9-e2ca-488d-839c-b402cb0d06d4";
    private const string DualAovPassGuid = "2687ad5d-7d89-43f6-a429-ae30087a31a5";
    private const string CombinedOutputGuid = "08e274e2-12c8-4746-8240-9cb7f4245342";
    private const string NormalsOutputGuid = "186db21d-bb6f-4b86-b85c-a4a431143176";
    private const string NormalsOutputSocketGuid = "03e1c601-7ac8-42c4-9897-305db6e8533a";

    [Fact]
    public void ProgressiveHistoryAccumulatesAndHonorsPreviewIntervals()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "temporal-manifest.json");
        fixture.WriteScene();
        WriteTemporalGraph(fixture, targetSamples: 4, previewEverySamples: 3);
        WriteManifest(manifestPath, AccumulatorManifest());
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var first = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(1, FrameRed(fixture));
        Assert.True(first.FramePublished);
        Assert.True(first.HistoryReset);

        var second = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(1, FrameRed(fixture));
        Assert.False(second.FramePublished);
        Assert.Equal(2, second.AccumulatedSamples);

        var third = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(3, FrameRed(fixture));
        Assert.True(third.FramePublished);

        var fourth = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(4, FrameRed(fixture));
        Assert.True(fourth.FramePublished);
        Assert.True(fourth.Completed);
        Assert.False(fourth.NeedsMoreWork);
    }

    [Fact]
    public void GraphAndCameraChangesResetProgressiveHistory()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "temporal-manifest.json");
        fixture.WriteScene();
        WriteTemporalGraph(fixture, increment: 1);
        WriteManifest(manifestPath, AccumulatorManifest());
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        _ = host.RenderOnce(fixture.RequestPath);
        _ = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(2, FrameRed(fixture));

        WriteTemporalGraph(fixture, increment: 4);
        var graphReset = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(4, FrameRed(fixture));
        Assert.True(graphReset.HistoryReset);

        var request = JsonNode.Parse(File.ReadAllText(fixture.RequestPath))!.AsObject();
        request["viewProjection"]![0] = 2.0f;
        File.WriteAllText(fixture.RequestPath, request.ToJsonString());
        var cameraReset = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(4, FrameRed(fixture));
        Assert.True(cameraReset.HistoryReset);
    }

    [Fact]
    public void ViewsKeepIndependentHistory()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "temporal-manifest.json");
        fixture.WriteScene();
        WriteManifest(manifestPath, AccumulatorManifest());
        WriteTemporalGraph(fixture, viewId: "view-1");
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        _ = host.RenderOnce(fixture.RequestPath);
        _ = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(2, FrameRed(fixture));

        WriteTemporalGraph(fixture, viewId: "view-2");
        SetRequestView(fixture, "view-2");
        _ = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(1, FrameRed(fixture));

        WriteTemporalGraph(fixture, viewId: "view-1");
        SetRequestView(fixture, "view-1");
        _ = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(3, FrameRed(fixture));
    }

    [Fact]
    public void FailedPassDoesNotPublishOrCommitHistory()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "temporal-manifest.json");
        var failurePath = Path.Combine(fixture.Root, "fail-pass");
        fixture.WriteScene();
        WriteTemporalGraph(fixture, failurePath: failurePath);
        WriteManifest(manifestPath, AccumulatorManifest(), FailureManifest());
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var first = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(1, FrameRed(fixture));
        Assert.Equal(1, first.AccumulatedSamples);

        File.WriteAllText(failurePath, "fail");
        Assert.Throws<InvalidOperationException>(() => host.RenderOnce(fixture.RequestPath));
        Assert.Equal(1, FrameRed(fixture));

        File.Delete(failurePath);
        var recovered = host.RenderOnce(fixture.RequestPath);
        Assert.Equal(2, FrameRed(fixture));
        Assert.Equal(2, recovered.AccumulatedSamples);
        Assert.False(recovered.HistoryReset);
    }

    [Fact]
    public void SelectedAovControlsThePublishedTextureAndResultMetadata()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "aov-manifest.json");
        fixture.WriteScene();
        WriteManifest(manifestPath, DualAovManifest());
        WriteAovGraph(fixture, RenderGraphDocument.OutputColorSocketGuid, "Combined");
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var combined = host.RenderOnce(fixture.RequestPath);
        Assert.Equal("Combined", combined.Aov);
        Assert.Equal(new byte[] { 220, 10, 20, 255 }, FramePixel(fixture));

        WriteAovGraph(fixture, NormalsOutputSocketGuid, "Normals");
        var normals = host.RenderOnce(fixture.RequestPath);
        Assert.Equal("Normals", normals.Aov);
        Assert.Equal(new byte[] { 20, 40, 230, 255 }, FramePixel(fixture));
    }

    [Fact]
    public void SelectedAovLabelResolvesAnAlternatePassOutputWithoutRewiringCombined()
    {
        using var fixture = new ProtocolFixture();
        var manifestPath = Path.Combine(fixture.Root, "aov-label-manifest.json");
        fixture.WriteScene();
        WriteManifest(manifestPath, DualAovManifest());
        WriteAovGraph(
            fixture,
            RenderGraphDocument.OutputColorSocketGuid,
            "AOV Pass \u00b7 Normals");
        fixture.WriteRequest(manifestPath: manifestPath);
        using var host = new RenderHostRunner();

        var normals = host.RenderOnce(fixture.RequestPath);

        Assert.Equal("AOV Pass \u00b7 Normals", normals.Aov);
        Assert.Equal(new byte[] { 20, 40, 230, 255 }, FramePixel(fixture));
    }

    [Fact]
    public async Task OfflineInvocationRunsToItsTargetWhileUnboundedProgressiveRunsOnce()
    {
        using var offline = new ProtocolFixture();
        var offlineManifest = Path.Combine(offline.Root, "temporal-manifest.json");
        offline.WriteScene();
        WriteTemporalGraph(
            offline,
            executionMode: "OFFLINE",
            targetSamples: 3,
            previewEverySamples: 100);
        WriteManifest(offlineManifest, AccumulatorManifest());
        offline.WriteRequest(manifestPath: offlineManifest);

        Assert.Equal(0, await RenderHostProgram.RunAsync(["--request", offline.RequestPath]));
        Assert.Equal(3, FrameRed(offline));

        using var progressive = new ProtocolFixture();
        var progressiveManifest = Path.Combine(progressive.Root, "temporal-manifest.json");
        progressive.WriteScene();
        WriteTemporalGraph(progressive, targetSamples: 0);
        WriteManifest(progressiveManifest, AccumulatorManifest());
        progressive.WriteRequest(manifestPath: progressiveManifest);

        Assert.Equal(0, await RenderHostProgram.RunAsync(["--request", progressive.RequestPath]));
        Assert.Equal(1, FrameRed(progressive));
    }

    [Fact]
    public void InvalidHistoryKeysAndLinksAreRejected()
    {
        using var fixture = new ProtocolFixture();
        WriteTemporalGraph(fixture);
        var graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        var historyRead = graph["nodes"]!.AsArray().Single(node =>
            node!["kind"]!.GetValue<string>() == "history-read")!.AsObject();
        historyRead.Remove("historyKey");
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());
        Assert.Contains(
            "historyKey",
            Assert.Throws<InvalidDataException>(
                () => RenderGraphDocument.Load(fixture.GraphPath)).Message);

        WriteTemporalGraph(fixture);
        graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        graph["links"]!.AsArray().Add(new JsonObject
        {
            ["fromNode"] = "scene",
            ["fromSocket"] = RenderGraphDocument.SceneGeometrySocketGuid,
            ["toNode"] = "history-read",
            ["toSocket"] = "bad-input"
        });
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());
        Assert.Contains(
            "cannot have inputs",
            Assert.Throws<InvalidDataException>(
                () => RenderGraphDocument.Load(fixture.GraphPath)).Message);

        WriteTemporalGraph(fixture);
        graph = JsonNode.Parse(File.ReadAllText(fixture.GraphPath))!.AsObject();
        var historyLink = graph["links"]!.AsArray().Single(node =>
            node!["toNode"]!.GetValue<string>() == "history-write")!.AsObject();
        historyLink["toSocket"] = "bad-current";
        File.WriteAllText(fixture.GraphPath, graph.ToJsonString());
        Assert.Contains(
            "exactly one Current input",
            Assert.Throws<InvalidDataException>(
                () => RenderGraphDocument.Load(fixture.GraphPath)).Message);
    }

    private static void WriteTemporalGraph(
        ProtocolFixture fixture,
        string executionMode = "PROGRESSIVE",
        int targetSamples = 0,
        int previewEverySamples = 1,
        int increment = 1,
        string viewId = "view-1",
        string? failurePath = null)
    {
        var nodes = new JsonArray(
            new JsonObject { ["nodeId"] = "scene", ["kind"] = "scene" },
            new JsonObject
            {
                ["nodeId"] = "history-read",
                ["kind"] = "history-read",
                ["historyKey"] = "accumulation"
            },
            new JsonObject
            {
                ["nodeId"] = "accumulator",
                ["kind"] = "pass",
                ["passGuid"] = AccumulatorPassGuid,
                ["typeName"] = typeof(IncrementHistoryPass).FullName,
                ["muted"] = false,
                ["parameters"] = new JsonObject { ["Increment"] = increment }
            },
            new JsonObject
            {
                ["nodeId"] = "history-write",
                ["kind"] = "history-write",
                ["historyKey"] = "accumulation"
            });
        var links = new JsonArray(
            Link("history-read", RenderGraphDocument.HistoryReadSocketGuid, "accumulator", AccumulatorInputGuid),
            Link("accumulator", AccumulatorOutputGuid, "history-write", RenderGraphDocument.HistoryWriteSocketGuid));
        var order = new JsonArray("scene", "history-read", "accumulator", "history-write");
        if (failurePath is not null)
        {
            nodes.Add(new JsonObject
            {
                ["nodeId"] = "failure",
                ["kind"] = "pass",
                ["passGuid"] = FailurePassGuid,
                ["typeName"] = typeof(FailAfterSubmissionPass).FullName,
                ["muted"] = false,
                ["parameters"] = new JsonObject { ["FailurePath"] = failurePath }
            });
            links.Add(Link("accumulator", AccumulatorOutputGuid, "failure", FailureInputGuid));
            order.Add("failure");
        }
        nodes.Add(new JsonObject { ["nodeId"] = "output", ["kind"] = "output" });
        links.Add(failurePath is null
            ? Link("accumulator", AccumulatorOutputGuid, "output", RenderGraphDocument.OutputColorSocketGuid)
            : Link("failure", FailureOutputGuid, "output", RenderGraphDocument.OutputColorSocketGuid));
        order.Add("output");

        WriteGraph(fixture, new JsonObject
        {
            ["schemaVersion"] = 1,
            ["generationId"] = ProtocolFixture.GenerationId,
            ["graphId"] = "temporal-graph",
            ["viewId"] = viewId,
            ["viewKind"] = "CUSTOM",
            ["executionMode"] = executionMode,
            ["resolutionScale"] = 1.0f,
            ["sampleCount"] = 1,
            ["targetSamples"] = targetSamples,
            ["samplesPerIteration"] = 1,
            ["previewEverySamples"] = previewEverySamples,
            ["nodes"] = nodes,
            ["links"] = links,
            ["topologicalOrder"] = order,
            ["output"] = new JsonObject
            {
                ["nodeId"] = "output",
                ["socketGuid"] = RenderGraphDocument.OutputColorSocketGuid,
                ["aov"] = "Combined"
            }
        });
    }

    private static void WriteAovGraph(
        ProtocolFixture fixture,
        string selectedSocket,
        string aov)
    {
        WriteGraph(fixture, new JsonObject
        {
            ["schemaVersion"] = 1,
            ["generationId"] = ProtocolFixture.GenerationId,
            ["graphId"] = "aov-graph",
            ["viewId"] = "view-1",
            ["viewKind"] = "CUSTOM",
            ["executionMode"] = "REALTIME",
            ["resolutionScale"] = 1.0f,
            ["sampleCount"] = 1,
            ["nodes"] = new JsonArray(
                new JsonObject { ["nodeId"] = "scene", ["kind"] = "scene" },
                new JsonObject
                {
                    ["nodeId"] = "aov-pass",
                    ["kind"] = "pass",
                    ["name"] = "AOV Pass",
                    ["displayName"] = "AOV Pass",
                    ["passGuid"] = DualAovPassGuid,
                    ["typeName"] = typeof(DualAovPass).FullName,
                    ["muted"] = false,
                    ["parameters"] = new JsonObject(),
                    ["outputs"] = new JsonArray(
                        new JsonObject
                        {
                            ["socketGuid"] = CombinedOutputGuid,
                            ["name"] = "Combined",
                            ["resourceKind"] = "Texture2D"
                        },
                        new JsonObject
                        {
                            ["socketGuid"] = NormalsOutputGuid,
                            ["name"] = "Normals",
                            ["resourceKind"] = "Texture2D"
                        })
                },
                new JsonObject { ["nodeId"] = "output", ["kind"] = "output" }),
            ["links"] = new JsonArray(
                Link("aov-pass", CombinedOutputGuid, "output", RenderGraphDocument.OutputColorSocketGuid),
                Link("aov-pass", NormalsOutputGuid, "output", NormalsOutputSocketGuid)),
            ["topologicalOrder"] = new JsonArray("scene", "aov-pass", "output"),
            ["output"] = new JsonObject
            {
                ["nodeId"] = "output",
                ["socketGuid"] = selectedSocket,
                ["aov"] = aov
            }
        });
    }

    private static JsonObject Link(string fromNode, string fromSocket, string toNode, string toSocket)
        => new()
        {
            ["fromNode"] = fromNode,
            ["fromSocket"] = fromSocket,
            ["toNode"] = toNode,
            ["toSocket"] = toSocket
        };

    private static void WriteGraph(ProtocolFixture fixture, JsonObject graph)
        => File.WriteAllText(
            fixture.GraphPath,
            graph.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);

    private static void SetRequestView(ProtocolFixture fixture, string viewId)
    {
        var request = JsonNode.Parse(File.ReadAllText(fixture.RequestPath))!.AsObject();
        request["viewId"] = viewId;
        File.WriteAllText(fixture.RequestPath, request.ToJsonString());
    }

    private static byte FrameRed(ProtocolFixture fixture) => FramePixel(fixture)[0];

    private static byte[] FramePixel(ProtocolFixture fixture)
        => File.ReadAllBytes(fixture.OutputPath)[40..44];

    private static void WriteManifest(string path, params ManifestPass[] passes)
    {
        var assemblyPath = typeof(TemporalRenderHostIntegrationTests).Assembly.Location;
        var document = new JsonObject
        {
            ["schemaVersion"] = 1,
            ["buildId"] = string.Empty,
            ["assemblyPath"] = assemblyPath,
            ["feirDirectory"] = "",
            ["projectRoot"] = ".",
            ["passes"] = new JsonArray(passes.Select(pass =>
                (JsonNode)new JsonObject
                {
                    ["passGuid"] = pass.PassGuid,
                    ["typeName"] = pass.PassType.FullName,
                    ["assemblyPath"] = assemblyPath,
                    ["inputs"] = new JsonArray(pass.Inputs.Select(SocketJson).ToArray()),
                    ["outputs"] = new JsonArray(pass.Outputs.Select(SocketJson).ToArray())
                }).ToArray())
        };
        var options = new JsonSerializerOptions { WriteIndented = true };
        var normalized = document.ToJsonString(options);
        using var hasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hasher.AppendData(Encoding.UTF8.GetBytes(normalized));
        hasher.AppendData([0]);
        hasher.AppendData(File.ReadAllBytes(assemblyPath));
        document["buildId"] = "sha256:" + Convert.ToHexString(
            hasher.GetHashAndReset()).ToLowerInvariant();
        File.WriteAllText(path, document.ToJsonString(options) + Environment.NewLine);
    }

    private static ManifestPass AccumulatorManifest()
        => new(
            typeof(IncrementHistoryPass),
            AccumulatorPassGuid,
            [new(AccumulatorInputGuid, "Texture2D", "RGBA8")],
            [new(AccumulatorOutputGuid, "Texture2D", "RGBA8")]);

    private static ManifestPass FailureManifest()
        => new(
            typeof(FailAfterSubmissionPass),
            FailurePassGuid,
            [new(FailureInputGuid, "Texture2D", "RGBA8")],
            [new(FailureOutputGuid, "Texture2D", "RGBA8")]);

    private static ManifestPass DualAovManifest()
        => new(
            typeof(DualAovPass),
            DualAovPassGuid,
            [],
            [
                new(CombinedOutputGuid, "Texture2D", "RGBA8"),
                new(NormalsOutputGuid, "Texture2D", "RGBA8")
            ]);

    private static JsonNode SocketJson(ManifestSocket socket)
        => new JsonObject
        {
            ["socketGuid"] = socket.SocketGuid,
            ["resourceKind"] = socket.ResourceKind,
            ["format"] = socket.Format
        };

    [FeatherPass(AccumulatorPassGuid)]
    public sealed class IncrementHistoryPass : IComputePass
    {
        [Input(AccumulatorInputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle History { get; init; }

        [Output(AccumulatorOutputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Output { get; init; }

        [Parameter(IncrementParameterGuid)]
        public int Increment { get; set; } = 1;

        public void Execute(RenderContext context)
        {
            var input = context.GetColorInput(History).Span;
            var output = new Rgba8[input.Length];
            for (var index = 0; index < input.Length; index++)
            {
                var pixel = input[index];
                output[index] = new Rgba8(
                    checked((byte)System.Math.Min(byte.MaxValue, pixel.R + Increment)),
                    pixel.G,
                    pixel.B,
                    pixel.A);
            }
            context.SetColorOutput(Output, output);
        }
    }

    [FeatherPass(FailurePassGuid)]
    public sealed class FailAfterSubmissionPass : IComputePass
    {
        [Input(FailureInputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Input { get; init; }

        [Output(FailureOutputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Output { get; init; }

        [Parameter(FailurePathParameterGuid)]
        public string FailurePath { get; set; } = "";

        public void Execute(RenderContext context)
        {
            context.SetColorOutput(Output, context.GetColorInput(Input).Span);
            if (File.Exists(FailurePath))
            {
                throw new InvalidOperationException("Requested failure after output submission.");
            }
        }
    }

    [FeatherPass(DualAovPassGuid)]
    public sealed class DualAovPass : IComputePass
    {
        [Output(CombinedOutputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Combined { get; init; }

        [Output(NormalsOutputGuid, Format = TextureFormat.Rgba8)]
        public TextureHandle Normals { get; init; }

        public void Execute(RenderContext context)
        {
            var combined = new Rgba8[checked(context.Width * context.Height)];
            var normals = new Rgba8[combined.Length];
            Array.Fill(combined, new Rgba8(220, 10, 20, 255));
            Array.Fill(normals, new Rgba8(20, 40, 230, 255));
            context.SetColorOutput(Combined, combined);
            context.SetColorOutput(Normals, normals);
        }
    }

    private sealed record ManifestPass(
        Type PassType,
        string PassGuid,
        ManifestSocket[] Inputs,
        ManifestSocket[] Outputs);

    private sealed record ManifestSocket(
        string SocketGuid,
        string ResourceKind,
        string Format);
}
