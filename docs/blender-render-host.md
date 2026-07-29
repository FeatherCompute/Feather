# Blender RenderHost

`Feather.Blender.RenderHost` is the out-of-process GPU renderer used by the
Feather Blender fork. The MVP consumes Blender's evaluated scene snapshot and a
validated pass graph, renders with the public Feather graphics API, reads the
RGBA8 target back to the CPU, and atomically publishes `viewport.frame`.

## SDK and tool distribution

Generated projects pin `FeatherCompute` and
`FeatherCompute.Blender.RenderHost` to the same version. Without a configured
source checkout, `dotnet build` restores the SDK package and Blender restores
the local RenderHost tool from `.config/dotnet-tools.json` on first use. This is
the normal machine-independent path.

Feather contributors can override the package with a source checkout in CLI or
Rider using either form below:

```bash
FEATHER_SDK_ROOT=/path/to/Feather dotnet build MyExperiment.csproj
dotnet build MyExperiment.csproj -p:FeatherSdkRoot=/path/to/Feather
```

Generated projects may also contain `.feather/local.props`. It is an ignored,
machine-local convenience file and uses a project-relative path when possible;
it must not be committed. The shared project file additionally recognizes a
valid `../Feather` checkout. An explicitly configured but invalid source path
fails early; with no source configuration the project uses NuGet. Set
`FeatherUseNuGet=true` to force package mode on a development machine.

## Run

Render one request:

```bash
dotnet run --project src/Feather.Blender.RenderHost -- \
  --request /path/to/project/.feather/cache/viewport.request.json
```

From a generated project using the packaged tool, the equivalent command is:

```bash
dotnet tool restore
dotnet tool run feather-blender-renderhost -- \
  --request .feather/cache/viewport.request.json
```

Keep the GPU process alive and render each atomically replaced request:

```bash
dotnet run --project src/Feather.Blender.RenderHost -- \
  --request /path/to/project/.feather/cache/viewport.request.json \
  --watch --poll-ms 33
```

The process writes `frame` events when a preview is published and `progress`
events for completed iterations that do not cross the preview interval. Errors
are written as JSON events to standard error. A failed pass never replaces the
last valid frame or commits temporal history.

## Request V1

Paths may be absolute or relative to the request file. Blender writes the
request to a temporary file and atomically replaces the published path.

```json
{
  "schemaVersion": 1,
  "requestId": 42,
  "generationId": "5ebc93da-b905-4f44-8eda-68968bb6ba2f",
  "viewId": "d7bb05b2-3bf8-4d75-8c74-f51d06feb91e",
  "width": 960,
  "height": 540,
  "matrixLayout": "row-major",
  "clipSpace": "blender-opengl",
  "viewProjection": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  "scenePath": "scene.featherscene",
  "graphPath": "graph.json",
  "manifestPath": "../../Generated/pass-manifest.json",
  "outputPath": "viewport.frame"
}
```

`viewProjection` is Blender's `RegionView3D.perspective_matrix`. The host
converts its OpenGL Y/depth convention to Vulkan. `clipSpace: "vulkan"` skips
that conversion and is useful for tests and non-Blender producers.

`generationId` is also stored in the graph and scene metadata. The host rejects
a frame unless all three values match, so independently replaced files from two
viewport updates cannot be combined into one render.

`manifestPath` selects the exported project pass manifest. The host validates the
manifest `buildId` against the assembly bytes, loads the assembly into a collectible
`AssemblyLoadContext`, and resolves the graph pass by stable GUID and C# type. A
new `buildId` is loaded before the previous context is unloaded, so a failed build
or load does not invalidate the last loaded generation. In watch mode, replacing
the manifest triggers a render even when the viewport request itself is unchanged.

Exported manifests include `projectRoot`, relative to the manifest directory, so
project-relative `assemblyPath` values remain valid when the whole project moves.
The authoritative execution artifact is currently the assembly. `feirPath` is
empty unless an independent FEIR artifact actually exists; generated shader IR is
embedded in the assembly.

## Graph V1

The graph document mirrors stable Blender node and socket identities. Blender
also writes a topological order after checking link types, duplicate input
links, and cycles.

```json
{
  "schemaVersion": 1,
  "generationId": "5ebc93da-b905-4f44-8eda-68968bb6ba2f",
  "graphId": "9fd54230-a114-4b20-a8c6-250217e6cfaa",
  "viewId": "d7bb05b2-3bf8-4d75-8c74-f51d06feb91e",
  "viewKind": "MATERIAL_PREVIEW",
  "executionMode": "REALTIME",
  "resolutionScale": 1.0,
  "sampleCount": 1,
  "targetSamples": 1,
  "samplesPerIteration": 1,
  "previewEverySamples": 1,
  "nodes": [
    { "nodeId": "scene", "kind": "scene", "name": "Scene", "muted": false },
    {
      "nodeId": "raster",
      "kind": "pass",
      "name": "Minimal Raster",
      "muted": false,
      "passGuid": "01c671a1-9b4e-5cab-b7e1-c101348af596",
      "typeName": "MyProject.Passes.MinimalRasterPass",
      "parameters": []
    },
    { "nodeId": "output", "kind": "output", "name": "Output", "muted": false }
  ],
  "links": [
    {
      "fromNode": "scene",
      "fromSocket": "b5db545a-ec06-557c-8b3e-2bc38c8193ef",
      "toNode": "raster",
      "toSocket": "6d6eb2d5-bb7a-55a4-a85a-c58e36715c53"
    },
    {
      "fromNode": "scene",
      "fromSocket": "f4fe7a75-0c26-56d1-af67-01ac7638fe16",
      "toNode": "raster",
      "toSocket": "a6eed590-b632-5f91-a69d-09b6eb4bb5ac"
    },
    {
      "fromNode": "scene",
      "fromSocket": "6078325d-ed5e-5aa7-a103-1b3292605c40",
      "toNode": "raster",
      "toSocket": "cc78191c-ac9a-57b6-bcac-91cce5e298f5"
    },
    {
      "fromNode": "raster",
      "fromSocket": "bd711ea6-36f9-56cd-863a-cfec58727a46",
      "toNode": "output",
      "toSocket": "082faef8-760d-5062-9766-2d627d8c42f8"
    }
  ],
  "topologicalOrder": ["scene", "raster", "output"],
  "output": {
    "nodeId": "output",
    "socketGuid": "082faef8-760d-5062-9766-2d627d8c42f8",
    "aov": "Combined"
  }
}
```

Project graphs may contain any number of manifest-defined raster and compute
passes. The host validates the exported topological order, required inputs,
resource kind/format compatibility, and the selected output link. Host-owned
RGBA8 textures carry intermediate results between passes. A muted pass is
bypassed only when it has one compatible connected Texture2D input.

For each unmuted node the host creates the project pass, binds handles by stable
socket GUID, converts instance values to `[Parameter]` members, executes it, and
disposes an `IDisposable` pass. Requests without `manifestPath` retain the
single-pass built-in MinimalRaster implementation only as a protocol
compatibility fallback.

### Scheduling, History, And AOV

`REALTIME` and `ON_DEMAND` execute once per request. `PROGRESSIVE` repeats in
watch mode; `targetSamples: 0` means unbounded. `OFFLINE` repeats in one host
invocation until `targetSamples` is reached. Missing or zero iteration/preview
values normalize to one, and an absent offline target preserves old behavior by
normalizing to one. The Feather View panel stores and exports these controls and
the selected AOV label with the `.blend`.

The first iteration, preview interval crossings, and the final iteration publish
a frame. Result events report `executionMode`, `aov`, `iteration`,
`accumulatedSamples`, `targetSamples`, `framePublished`, `completed`,
`needsMoreWork`, `historyReset`, and `resetCount`. Non-watch unbounded
progressive execution intentionally renders one iteration and exits successfully.

Frame and progress events also expose stage timings in milliseconds:
`protocolLoadMilliseconds`, `sceneLoadMilliseconds`,
`sceneBuildMilliseconds`, `passExecutionMilliseconds`,
`gpuReadbackMilliseconds`, `frameWriteMilliseconds`, and
`totalMilliseconds`. `passExecutionMilliseconds` includes synchronous GPU
readback; `gpuReadbackMilliseconds` is the measured `GpuTexture2D.Read` subset,
not an additional stage to add to it. CPU-authored pass outputs report zero
readback time.

Viewport frame publication closes the temporary file and atomically replaces
the previous frame. It intentionally does not request an `fsync`: the file is
an ephemeral IPC snapshot, not durable project data, and the atomic rename is
the consistency boundary Blender needs.

Temporal resources remain outside the ordinary DAG:

```json
{
  "nodeId": "history-read",
  "kind": "history-read",
  "historyKey": "taa-color"
}
```

History Read uses output socket
`b85a7129-ad17-5d67-b06b-60e15ce071d0`; History Write consumes one Texture2D at
`8d513f8b-7212-557b-bcec-2f88ed212c21`. Matching `historyKey` values connect
frames implicitly. The first read is opaque black, state is isolated by View,
and writes commit only after every pass and the selected output succeed.

Accumulation resets when generation, graph contents, dimensions, camera,
selected output/AOV, scheduling configuration, or project assembly changes.
The graph-content identity is a SHA-256 of the published graph document, so a
parameter or link edit resets history even when the persistent `graphId` stays
the same. `output.aov` names the single selected output; changing
`output.socketGuid` genuinely selects a different linked texture.

## Public Pass Contract

Project code receives only public Feather APIs. `RenderContext` exposes the
requested dimensions/MSAA count, immutable scene geometry, material/texture/light
tables, timeline position, camera, and RGBA8 graph inputs. A pass defines its own
GPU layout and creates buffers, textures, shaders, compute kernels, and pipelines
through `GPU`. It publishes each output with:

```csharp
var scene = context.GetSceneGeometry(Geometry);
var materials = context.GetMaterials(Materials);
var textures = context.GetTextures(Textures);
var lights = context.GetLights(Lights);
var time = context.GetTime(Time);
var camera = context.GetCamera(Camera);
context.SetColorOutput(Color, colorTexture, pipeline.LastDispatchPath);
```

`SetColorOutput` performs the synchronous RGBA8 readback used by the current
viewport bridge. A CPU-span overload is also available for software renderers.
The host rejects a pass that does not submit its selected Color output.

## Scene And Frame

The scene file starts with `FTHSCN01`, schema version 2, JSON metadata length,
and payload length. Its payload contains little-endian `float32`, `uint32`, and
RGBA8 `uint8` arrays described by byte offsets and shapes. Version 1 remains
readable.

Version 2 carries evaluated positions, corner normals/UVs, triangle material
indices, object instances, a material table, image pixels and hashes, lights,
and frame/subframe. The current Blender translator supports non-node materials
and a strict Material Output -> Principled BSDF subset with optional Image
Texture using active UVs. Unsupported graphs become an explicit magenta fallback
with a diagnostic instead of being silently approximated. The default public
MinimalRaster pass consumes those resources for white, normal-debug, and basic
material-preview Views.

The output uses Blender's `FTHRFRM1` 40-byte header. The host publishes tightly
packed RGBA8 rows with top-left origin, followed by `width * height * 4` bytes.
Blender normalizes the origin before uploading the frame to its GPU texture.

## Current Boundaries

- Intermediate and history graph textures are RGBA8 CPU frames; arbitrary GPU
  resource kinds and formats are later extensions.
- Material translation intentionally covers only the documented Principled and
  Image Texture subset. It is not Eevee/Cycles compatibility.
- The default preview uses a small direct-lighting model, one selected Sun or
  Point light, and a fixed sampler. More BSDF and sampler semantics are coverage
  work, not hidden compatibility behavior.
- Meshes and instances are rebuilt per request. There is no incremental GPU
  scene cache yet.
- Readback is synchronous by design for the first integration. Async staging
  rings can be added after measured viewport data justifies them.
