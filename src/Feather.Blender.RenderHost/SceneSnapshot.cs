using System.Buffers.Binary;
using System.Text.Json;
using Feather.Math;

namespace Feather.Blender.RenderHost;

internal sealed record SceneSnapshot(SceneMetadata Metadata, byte[] Payload)
{
    private static ReadOnlySpan<byte> Magic => "FTHSCN01"u8;
    private const int HeaderSize = 24;
    private const int CurrentSchemaVersion = 1;
    private const int MaximumMetadataLength = 64 * 1024 * 1024;
    private const int MaximumPayloadLength = 1024 * 1024 * 1024;

    public static SceneSnapshot Load(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
        if (stream.Length < HeaderSize)
        {
            throw new InvalidDataException("Scene snapshot header is truncated.");
        }

        Span<byte> header = stackalloc byte[HeaderSize];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(Magic))
        {
            throw new InvalidDataException("Scene snapshot magic is invalid.");
        }

        var version = BinaryPrimitives.ReadUInt32LittleEndian(header[8..12]);
        var metadataLength = BinaryPrimitives.ReadUInt32LittleEndian(header[12..16]);
        var payloadLength = BinaryPrimitives.ReadUInt64LittleEndian(header[16..24]);
        if (version != CurrentSchemaVersion)
        {
            throw new InvalidDataException($"Unsupported scene snapshot version: {version}.");
        }
        if (metadataLength > MaximumMetadataLength)
        {
            throw new InvalidDataException("Scene snapshot metadata exceeds the host limit.");
        }
        if (payloadLength > MaximumPayloadLength)
        {
            throw new InvalidDataException("Scene snapshot payload exceeds the host limit.");
        }

        var expectedLength = checked(HeaderSize + (long)metadataLength + (long)payloadLength);
        if (stream.Length != expectedLength)
        {
            throw new InvalidDataException(
                $"Scene snapshot length is {stream.Length} bytes; expected {expectedLength} bytes.");
        }

        var metadataBytes = new byte[checked((int)metadataLength)];
        var payload = new byte[checked((int)payloadLength)];
        stream.ReadExactly(metadataBytes);
        stream.ReadExactly(payload);

        SceneMetadata metadata;
        try
        {
            metadata = JsonSerializer.Deserialize<SceneMetadata>(metadataBytes, ProtocolJson.Options)
                ?? throw new InvalidDataException("Scene snapshot metadata contains null.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"Scene snapshot metadata JSON is invalid: {exception.Message}", exception);
        }

        if (metadata.SchemaVersion != CurrentSchemaVersion)
        {
            throw new InvalidDataException(
                $"Scene metadata schema version {metadata.SchemaVersion} does not match the file header.");
        }
        if (!Guid.TryParse(metadata.GenerationId, out _))
        {
            throw new InvalidDataException("Scene metadata generationId must be a GUID.");
        }
        if (!string.Equals(metadata.MatrixLayout, "row-major", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Unsupported scene matrix layout: '{metadata.MatrixLayout}'.");
        }

        return new SceneSnapshot(metadata, payload);
    }

    public float[] ReadFloat32(ArrayDescriptor descriptor, string name, params int[] expectedShape)
    {
        ValidateDescriptor(descriptor, name, "float32", expectedShape);
        var result = new float[descriptor.ByteLength / sizeof(float)];
        var span = Payload.AsSpan(checked((int)descriptor.Offset), descriptor.ByteLength);
        for (var index = 0; index < result.Length; index++)
        {
            result[index] = BitConverter.Int32BitsToSingle(
                BinaryPrimitives.ReadInt32LittleEndian(span.Slice(index * sizeof(float), sizeof(float))));
            if (!float.IsFinite(result[index]))
            {
                throw new InvalidDataException($"Scene array '{name}' contains a non-finite value.");
            }
        }
        return result;
    }

    public uint[] ReadUInt32(ArrayDescriptor descriptor, string name, params int[] expectedShape)
    {
        ValidateDescriptor(descriptor, name, "uint32", expectedShape);
        var result = new uint[descriptor.ByteLength / sizeof(uint)];
        var span = Payload.AsSpan(checked((int)descriptor.Offset), descriptor.ByteLength);
        for (var index = 0; index < result.Length; index++)
        {
            result[index] = BinaryPrimitives.ReadUInt32LittleEndian(
                span.Slice(index * sizeof(uint), sizeof(uint)));
        }
        return result;
    }

    private void ValidateDescriptor(
        ArrayDescriptor descriptor,
        string name,
        string componentType,
        IReadOnlyList<int> expectedShape)
    {
        if (descriptor is null)
        {
            throw new InvalidDataException($"Scene array '{name}' is missing.");
        }
        if (!string.Equals(descriptor.ComponentType, componentType, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Scene array '{name}' must use {componentType} components.");
        }
        if (!descriptor.Shape.SequenceEqual(expectedShape))
        {
            throw new InvalidDataException(
                $"Scene array '{name}' shape [{string.Join(',', descriptor.Shape)}] does not match " +
                $"[{string.Join(',', expectedShape)}].");
        }

        long elementCount = 1;
        foreach (var dimension in expectedShape)
        {
            if (dimension < 0)
            {
                throw new InvalidDataException($"Scene array '{name}' has a negative dimension.");
            }
            elementCount = checked(elementCount * dimension);
        }
        var expectedLength = checked(elementCount * sizeof(uint));
        if (descriptor.ByteLength != expectedLength)
        {
            throw new InvalidDataException(
                $"Scene array '{name}' is {descriptor.ByteLength} bytes; expected {expectedLength} bytes.");
        }
        if (descriptor.Offset < 0 || descriptor.ByteLength < 0 ||
            descriptor.Offset > Payload.LongLength - descriptor.ByteLength)
        {
            throw new InvalidDataException($"Scene array '{name}' lies outside the snapshot payload.");
        }
    }
}

internal sealed class SceneMetadata
{
    public int SchemaVersion { get; init; }
    public string GenerationId { get; init; } = "";
    public string MatrixLayout { get; init; } = "";
    public int Frame { get; init; }
    public float Subframe { get; init; }
    public SceneMesh[] Meshes { get; init; } = [];
    public SceneInstance[] Instances { get; init; } = [];
}

internal sealed class SceneMesh
{
    public string MeshId { get; init; } = "";
    public string Name { get; init; } = "";
    public int VertexCount { get; init; }
    public int CornerCount { get; init; }
    public int TriangleCount { get; init; }
    public SceneMeshAttributes Attributes { get; init; } = new();
}

internal sealed class SceneMeshAttributes
{
    public ArrayDescriptor Positions { get; init; } = null!;
    public ArrayDescriptor LoopVertexIndices { get; init; } = null!;
    public ArrayDescriptor CornerNormals { get; init; } = null!;
    public ArrayDescriptor TriangleLoopIndices { get; init; } = null!;
}

internal sealed class ArrayDescriptor
{
    public long Offset { get; init; }
    public int ByteLength { get; init; }
    public string ComponentType { get; init; } = "";
    public int[] Shape { get; init; } = [];
}

internal sealed class SceneInstance
{
    public string InstanceId { get; init; } = "";
    public string Name { get; init; } = "";
    public string MeshId { get; init; } = "";
    public float[] MatrixWorld { get; init; } = [];
    public bool IsInstance { get; init; }
}

internal sealed record RenderGeometry(MinimalRasterVertex[] Vertices, uint[] Indices);

internal static class SceneGeometryBuilder
{
    public static RenderGeometry Build(SceneSnapshot snapshot)
    {
        var meshes = new Dictionary<string, ParsedMesh>(StringComparer.Ordinal);
        foreach (var mesh in snapshot.Metadata.Meshes)
        {
            if (string.IsNullOrWhiteSpace(mesh.MeshId) || !meshes.TryAdd(mesh.MeshId, ParseMesh(snapshot, mesh)))
            {
                throw new InvalidDataException($"Scene contains a missing or duplicate mesh ID '{mesh.MeshId}'.");
            }
        }

        var vertices = new List<MinimalRasterVertex>();
        var indices = new List<uint>();
        foreach (var instance in snapshot.Metadata.Instances)
        {
            if (!meshes.TryGetValue(instance.MeshId, out var mesh))
            {
                throw new InvalidDataException(
                    $"Scene instance '{instance.InstanceId}' references unknown mesh '{instance.MeshId}'.");
            }
            if (instance.MatrixWorld.Length != 16 || instance.MatrixWorld.Any(value => !float.IsFinite(value)))
            {
                throw new InvalidDataException($"Scene instance '{instance.InstanceId}' has an invalid matrixWorld.");
            }

            var model = MatrixProtocol.FromRowMajor(instance.MatrixWorld);
            float4x4 normalMatrix;
            try
            {
                normalMatrix = model.Inverse().Transposed();
            }
            catch (InvalidOperationException)
            {
                // A zero-scale Blender object is valid and usually invisible. Preserve its
                // transformed positions while using the model directions for a stable fallback normal.
                normalMatrix = model;
            }

            if ((long)vertices.Count + mesh.CornerCount > int.MaxValue)
            {
                throw new InvalidDataException("Scene geometry exceeds the managed GPU buffer limit.");
            }
            var baseVertex = checked((uint)vertices.Count);
            for (var corner = 0; corner < mesh.CornerCount; corner++)
            {
                var sourceVertex = mesh.LoopVertexIndices[corner];
                if (sourceVertex >= mesh.VertexCount)
                {
                    throw new InvalidDataException($"Mesh '{instance.MeshId}' loop references an invalid vertex.");
                }

                var positionOffset = checked((int)sourceVertex * 3);
                var normalOffset = corner * 3;
                var position = model * new float4(
                    mesh.Positions[positionOffset],
                    mesh.Positions[positionOffset + 1],
                    mesh.Positions[positionOffset + 2],
                    1.0f);
                if (MathF.Abs(position.W) > 1e-8f && position.W != 1.0f)
                {
                    position = position / position.W;
                }
                var normal4 = normalMatrix * new float4(
                    mesh.CornerNormals[normalOffset],
                    mesh.CornerNormals[normalOffset + 1],
                    mesh.CornerNormals[normalOffset + 2],
                    0.0f);
                var normal = Normalize(new float3(normal4.X, normal4.Y, normal4.Z));
                vertices.Add(new MinimalRasterVertex
                {
                    Position = new float3(position.X, position.Y, position.Z),
                    Normal = normal
                });
            }

            foreach (var loopIndex in mesh.TriangleLoopIndices)
            {
                if (loopIndex >= mesh.CornerCount)
                {
                    throw new InvalidDataException($"Mesh '{instance.MeshId}' triangle references an invalid loop.");
                }
                indices.Add(checked(baseVertex + loopIndex));
            }
        }

        return new RenderGeometry(vertices.ToArray(), indices.ToArray());
    }

    private static ParsedMesh ParseMesh(SceneSnapshot snapshot, SceneMesh mesh)
    {
        if (mesh.VertexCount < 0 || mesh.CornerCount < 0 || mesh.TriangleCount < 0)
        {
            throw new InvalidDataException($"Mesh '{mesh.MeshId}' has a negative element count.");
        }
        return new ParsedMesh(
            mesh.VertexCount,
            mesh.CornerCount,
            snapshot.ReadFloat32(mesh.Attributes.Positions, $"{mesh.MeshId}.positions", mesh.VertexCount, 3),
            snapshot.ReadUInt32(mesh.Attributes.LoopVertexIndices, $"{mesh.MeshId}.loopVertexIndices", mesh.CornerCount),
            snapshot.ReadFloat32(mesh.Attributes.CornerNormals, $"{mesh.MeshId}.cornerNormals", mesh.CornerCount, 3),
            snapshot.ReadUInt32(mesh.Attributes.TriangleLoopIndices, $"{mesh.MeshId}.triangleLoopIndices", mesh.TriangleCount, 3));
    }

    private static float3 Normalize(float3 value)
    {
        var length = MathF.Sqrt((value.X * value.X) + (value.Y * value.Y) + (value.Z * value.Z));
        return length > 1e-8f
            ? new float3(value.X / length, value.Y / length, value.Z / length)
            : new float3(0.0f, 0.0f, 1.0f);
    }

    private sealed record ParsedMesh(
        int VertexCount,
        int CornerCount,
        float[] Positions,
        uint[] LoopVertexIndices,
        float[] CornerNormals,
        uint[] TriangleLoopIndices);
}
