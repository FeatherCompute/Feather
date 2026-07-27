# Blender RenderHost

`Feather.Blender.RenderHost` is the out-of-process GPU renderer used by the
Feather Blender fork. The MVP consumes Blender's evaluated scene snapshot and a
validated pass graph, renders with the public Feather graphics API, reads the
RGBA8 target back to the CPU, and atomically publishes `viewport.frame`.

## Run

Render one request:

```bash
dotnet run --project src/Feather.Blender.RenderHost -- \
  --request /path/to/project/.feather/cache/viewport.request.json
```

Keep the GPU process alive and render each atomically replaced request:

```bash
dotnet run --project src/Feather.Blender.RenderHost -- \
  --request /path/to/project/.feather/cache/viewport.request.json \
  --watch --poll-ms 33
```

The process writes one JSON event per completed frame to standard output and
one JSON error event to standard error. A failed request never replaces the
last valid frame.

## Request V1

Paths may be absolute or relative to the request file. Blender writes the
request to a temporary file and atomically replaces the published path.

```json
{
  "schemaVersion": 1,
  "requestId": 42,
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

`manifestPath` is reserved for dynamic pass assembly loading. The MinimalRaster
MVP validates the graph's stable pass GUID and does not load user assemblies yet.

## Graph V1

The graph document mirrors stable Blender node and socket identities. Blender
also writes a topological order after checking link types, duplicate input
links, and cycles.

```json
{
  "schemaVersion": 1,
  "graphId": "9fd54230-a114-4b20-a8c6-250217e6cfaa",
  "viewId": "d7bb05b2-3bf8-4d75-8c74-f51d06feb91e",
  "executionMode": "REALTIME",
  "resolutionScale": 1.0,
  "sampleCount": 1,
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
      "fromNode": "raster",
      "fromSocket": "bd711ea6-36f9-56cd-863a-cfec58727a46",
      "toNode": "output",
      "toSocket": "082faef8-760d-5062-9766-2d627d8c42f8"
    }
  ],
  "topologicalOrder": ["scene", "raster", "output"],
  "output": {
    "nodeId": "output",
    "socketGuid": "082faef8-760d-5062-9766-2d627d8c42f8"
  }
}
```

The MVP accepts exactly one unmuted pass with the MinimalRaster GUID. Pass
parameters may be an object containing instance values or the current manifest
parameter-definition array. Supported MinimalRaster values are `clearColor`,
`lightDirection`, and `ambient`.

## Scene And Frame

The scene file starts with `FTHSCN01`, schema version 1, JSON metadata length,
and payload length. Its payload contains little-endian `float32` and `uint32`
arrays described by byte offsets and shapes. The host consumes evaluated mesh
positions, loop-to-vertex indices, corner normals, triangle loop indices, and
instance `matrixWorld` values. Corner vertices are transformed on the CPU and
drawn through a public Feather indexed graphics pipeline with a depth target.

The output uses Blender's `FTHRFRM1` 40-byte header. The host publishes tightly
packed RGBA8 rows with top-left origin, followed by `width * height * 4` bytes.
Blender normalizes the origin before uploading the frame to its GPU texture.

## MVP Boundaries

- Rendering is the built-in public-API MinimalRaster pass; dynamic project pass
  assembly loading and `RenderContext` resource binding remain to be added.
- Materials, textures, lights, UVs, and Material Nodes are present in or planned
  for the scene protocol but are not consumed by MinimalRaster.
- Meshes and instances are rebuilt per request. There is no incremental GPU
  scene cache yet.
- Readback is synchronous by design for the first integration. Async staging
  rings can be added after measured viewport data justifies them.
