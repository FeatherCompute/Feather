using System.Buffers.Binary;
using System.Text.Json;
using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost.Tests;

public sealed class SceneResourceTests
{
    [Fact]
    public void BlenderSceneV2ExposesUvMaterialsTexturesLightsAndTime()
    {
        using var fixture = new SceneV2Fixture();
        fixture.Write();

        var resources = SceneResourceBuilder.Build(SceneSnapshot.Load(fixture.Path));

        Assert.Equal(new float2(0.25f, 0.75f), resources.Geometry.Vertices[0].UV);
        Assert.Equal([0u, 1u, 2u], resources.Geometry.Indices);
        Assert.Equal(new SceneSubmesh(0, 3, 0), Assert.Single(resources.Geometry.Submeshes));

        var material = resources.Materials.Materials.Span[0];
        Assert.Equal("material-0", material.Id);
        Assert.Equal(new float4(0.2f, 0.3f, 0.4f, 0.8f), material.BaseColor);
        Assert.Equal(0.35f, material.Metallic);
        Assert.Equal(0.6f, material.Roughness);
        Assert.Equal(new float4(0.25f, 0.5f, 0.75f, 1.0f), material.EmissionColor);
        Assert.Equal(2.5f, material.EmissionStrength);
        Assert.Equal(0, material.BaseColorTextureIndex);
        Assert.Equal(SceneMaterialStatus.Supported, material.Status);

        var texture = Assert.Single(resources.Textures.Textures.ToArray());
        Assert.Equal((2, 1), (texture.Width, texture.Height));
        Assert.Equal("bottom-left", texture.Origin);
        Assert.Equal("rgba8-unorm", texture.Format);
        Assert.True(texture.Packed);
        Assert.Equal(new Rgba8(10, 20, 30, 255), texture.Pixels.Span[0]);
        Assert.Equal(new Rgba8(40, 50, 60, 128), texture.Pixels.Span[1]);

        var light = Assert.Single(resources.Lights.Lights.ToArray());
        Assert.Equal("light-0", light.Id);
        Assert.Equal(SceneLightType.Directional, light.Type);
        Assert.Equal(new float3(3.0f, 4.0f, 5.0f), light.Position);
        Assert.Equal(new float3(0.0f, 0.0f, -1.0f), light.Direction);
        Assert.Equal(new RenderTime(12, 0.25f), resources.Time);
    }

    [Fact]
    public void UnsupportedMaterialUsesExplicitMagentaFallbackAndDiagnostic()
    {
        using var fixture = new SceneV2Fixture();
        fixture.Write(graphStatus: "fallback", diagnostic: "UNSUPPORTED_SURFACE_NODE: Diffuse");

        var resources = SceneResourceBuilder.Build(SceneSnapshot.Load(fixture.Path));
        var material = resources.Materials.Materials.Span[0];

        Assert.Equal(SceneMaterialStatus.Fallback, material.Status);
        Assert.Equal(SceneMaterial.FallbackBaseColor, material.BaseColor);
        Assert.Equal(SceneMaterial.NoTexture, material.BaseColorTextureIndex);
        Assert.Equal("UNSUPPORTED_SURFACE_NODE: Diffuse", material.Diagnostic);
    }

    [Fact]
    public void BlenderSceneV1LegacyMaterialRemainsSupported()
    {
        using var fixture = new SceneV2Fixture();
        fixture.WriteLegacyV1();

        var resources = SceneResourceBuilder.Build(SceneSnapshot.Load(fixture.Path));
        var material = resources.Materials.Materials.Span[0];

        Assert.Equal(SceneMaterialStatus.Supported, material.Status);
        Assert.Equal(new float4(0.6f, 0.5f, 0.4f, 1.0f), material.BaseColor);
        Assert.Equal(0.2f, material.Metallic);
        Assert.Equal(0.7f, material.Roughness);
        Assert.Empty(resources.Textures.Textures.ToArray());
        Assert.Equal(new SceneSubmesh(0, 3, 0), Assert.Single(resources.Geometry.Submeshes));
    }

    [Fact]
    public void TextureDescriptorOutsidePayloadIsRejected()
    {
        using var fixture = new SceneV2Fixture();
        fixture.Write(invalidTextureOffset: true);

        var snapshot = SceneSnapshot.Load(fixture.Path);
        var exception = Assert.Throws<InvalidDataException>(() => SceneResourceBuilder.Build(snapshot));

        Assert.Contains("outside the snapshot payload", exception.Message);
    }

    private sealed class SceneV2Fixture : IDisposable
    {
        private readonly string root = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            $"feather-scene-v2-tests-{Guid.NewGuid():N}");

        public SceneV2Fixture()
        {
            Directory.CreateDirectory(root);
        }

        public string Path => System.IO.Path.Combine(root, "scene.featherscene");

        public void Write(
            string graphStatus = "supported",
            string diagnostic = "",
            bool invalidTextureOffset = false)
        {
            using var payload = new MemoryStream();
            var positions = WriteFloatArray(
                payload,
                [-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f],
                [3, 3]);
            var loopVertices = WriteUIntArray(payload, [0, 1, 2], [3]);
            var normals = WriteFloatArray(
                payload,
                [0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f],
                [3, 3]);
            var uvs = WriteFloatArray(payload, [0.25f, 0.75f, 1.0f, 0.0f, 0.0f, 1.0f], [3, 2]);
            var triangleLoops = WriteUIntArray(payload, [0, 1, 2], [1, 3]);
            var triangleMaterials = WriteUIntArray(payload, [0], [1]);
            var pixels = WriteByteArray(payload, [10, 20, 30, 255, 40, 50, 60, 128], [1, 2, 4]);
            if (invalidTextureOffset)
            {
                pixels["offset"] = payload.Length + 1;
            }

            var metadata = new
            {
                schemaVersion = 2,
                generationId = ProtocolFixture.GenerationId,
                matrixLayout = "row-major",
                frame = 12,
                subframe = 0.25f,
                meshes = new[]
                {
                    new
                    {
                        meshId = "mesh-0",
                        name = "Triangle",
                        vertexCount = 3,
                        cornerCount = 3,
                        triangleCount = 1,
                        activeUvName = "UVMap",
                        materialSlots = new string?[] { "material-0" },
                        attributes = new
                        {
                            positions,
                            loopVertexIndices = loopVertices,
                            cornerNormals = normals,
                            cornerUvs = uvs,
                            triangleLoopIndices = triangleLoops,
                            triangleMaterialIndices = triangleMaterials
                        }
                    }
                },
                instances = new[]
                {
                    new
                    {
                        instanceId = "instance-0",
                        name = "Triangle",
                        meshId = "mesh-0",
                        matrixWorld = IdentityMatrix(),
                        isInstance = false
                    }
                },
                materials = new[]
                {
                    new
                    {
                        materialId = "material-0",
                        name = "Paint",
                        diffuseColor = new[] { 0.2f, 0.3f, 0.4f, 0.8f },
                        baseColor = new[] { 0.2f, 0.3f, 0.4f, 0.8f },
                        metallic = 0.35f,
                        roughness = 0.6f,
                        emissionColor = new[] { 0.25f, 0.5f, 0.75f, 1.0f },
                        emissionStrength = 2.5f,
                        alpha = 0.8f,
                        baseColorTextureId = "texture-0",
                        graphStatus,
                        diagnostic,
                        useNodes = true,
                        nodeTree = "Paint Nodes"
                    }
                },
                textures = new[]
                {
                    new
                    {
                        textureId = "texture-0",
                        name = "Checker",
                        width = 2,
                        height = 1,
                        channels = 4,
                        componentType = "uint8",
                        format = "rgba8-unorm",
                        origin = "bottom-left",
                        colorSpace = "sRGB",
                        isData = false,
                        alphaMode = "STRAIGHT",
                        source = "GENERATED",
                        packed = true,
                        contentHash = new string('a', 64),
                        pixels
                    }
                },
                lights = new[]
                {
                    new
                    {
                        lightId = "light-0",
                        name = "Sun",
                        type = "SUN",
                        matrixWorld = new float[]
                        {
                            1, 0, 0, 3,
                            0, 1, 0, 4,
                            0, 0, 1, 5,
                            0, 0, 0, 1
                        },
                        position = new[] { 3.0f, 4.0f, 5.0f },
                        direction = new[] { 0.0f, 0.0f, -1.0f },
                        color = new[] { 1.0f, 0.9f, 0.8f },
                        energy = 4.0f,
                        radius = 0.1f,
                        spotSize = 0.0f,
                        spotBlend = 0.0f,
                        areaShape = (string?)null,
                        areaSize = 0.0f,
                        areaSizeY = 0.0f
                    }
                },
                camera = (object?)null
            };

            WriteFile(metadata, payload.ToArray(), 2);
        }

        public void WriteLegacyV1()
        {
            using var payload = new MemoryStream();
            var positions = WriteFloatArray(
                payload,
                [-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f],
                [3, 3]);
            var loopVertices = WriteUIntArray(payload, [0, 1, 2], [3]);
            var normals = WriteFloatArray(
                payload,
                [0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f],
                [3, 3]);
            var triangleLoops = WriteUIntArray(payload, [0, 1, 2], [1, 3]);
            var triangleMaterials = WriteUIntArray(payload, [0], [1]);
            var metadata = new
            {
                schemaVersion = 1,
                generationId = ProtocolFixture.GenerationId,
                matrixLayout = "row-major",
                frame = 2,
                subframe = 0.0f,
                meshes = new[]
                {
                    new
                    {
                        meshId = "mesh-0",
                        name = "Legacy Triangle",
                        vertexCount = 3,
                        cornerCount = 3,
                        triangleCount = 1,
                        materialSlots = new string?[] { "material-0" },
                        attributes = new
                        {
                            positions,
                            loopVertexIndices = loopVertices,
                            cornerNormals = normals,
                            triangleLoopIndices = triangleLoops,
                            triangleMaterialIndices = triangleMaterials
                        }
                    }
                },
                instances = new[]
                {
                    new
                    {
                        instanceId = "instance-0",
                        name = "Legacy Triangle",
                        meshId = "mesh-0",
                        matrixWorld = IdentityMatrix(),
                        isInstance = false
                    }
                },
                materials = new[]
                {
                    new
                    {
                        materialId = "material-0",
                        name = "Legacy",
                        diffuseColor = new[] { 0.6f, 0.5f, 0.4f, 1.0f },
                        metallic = 0.2f,
                        roughness = 0.7f,
                        useNodes = true,
                        nodeTree = "Legacy Nodes"
                    }
                },
                lights = Array.Empty<object>(),
                camera = (object?)null
            };
            WriteFile(metadata, payload.ToArray(), 1);
        }

        private void WriteFile(object metadata, byte[] payloadBytes, uint version)
        {
            var metadataBytes = JsonSerializer.SerializeToUtf8Bytes(metadata);
            using var file = File.Create(Path);
            Span<byte> header = stackalloc byte[24];
            "FTHSCN01"u8.CopyTo(header);
            BinaryPrimitives.WriteUInt32LittleEndian(header[8..12], version);
            BinaryPrimitives.WriteUInt32LittleEndian(header[12..16], checked((uint)metadataBytes.Length));
            BinaryPrimitives.WriteUInt64LittleEndian(header[16..24], checked((ulong)payloadBytes.Length));
            file.Write(header);
            file.Write(metadataBytes);
            file.Write(payloadBytes);
        }

        public void Dispose()
        {
            try
            {
                Directory.Delete(root, recursive: true);
            }
            catch (IOException)
            {
            }
        }

        private static float[] IdentityMatrix()
            =>
            [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            ];

        private static Dictionary<string, object> WriteFloatArray(
            Stream payload,
            IReadOnlyList<float> values,
            int[] shape)
        {
            var offset = payload.Position;
            Span<byte> bytes = stackalloc byte[sizeof(float)];
            foreach (var value in values)
            {
                BinaryPrimitives.WriteInt32LittleEndian(bytes, BitConverter.SingleToInt32Bits(value));
                payload.Write(bytes);
            }
            return Descriptor(offset, values.Count * sizeof(float), "float32", shape);
        }

        private static Dictionary<string, object> WriteUIntArray(
            Stream payload,
            IReadOnlyList<uint> values,
            int[] shape)
        {
            var offset = payload.Position;
            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            foreach (var value in values)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
                payload.Write(bytes);
            }
            return Descriptor(offset, values.Count * sizeof(uint), "uint32", shape);
        }

        private static Dictionary<string, object> WriteByteArray(
            Stream payload,
            IReadOnlyList<byte> values,
            int[] shape)
        {
            var offset = payload.Position;
            foreach (var value in values)
            {
                payload.WriteByte(value);
            }
            return Descriptor(offset, values.Count, "uint8", shape);
        }

        private static Dictionary<string, object> Descriptor(
            long offset,
            int byteLength,
            string componentType,
            int[] shape)
            => new()
            {
                ["offset"] = offset,
                ["byteLength"] = byteLength,
                ["componentType"] = componentType,
                ["shape"] = shape
            };
    }
}
