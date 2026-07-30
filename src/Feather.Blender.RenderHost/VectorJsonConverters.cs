using System.Text.Json;
using System.Text.Json.Serialization;
using Feather.Math;

namespace Feather.Blender.RenderHost;

/// <summary>
/// Reads Feather's vector types from JSON arrays.
///
/// Blender writes a vector pass parameter as <c>[x, y, z]</c>, because that is what its
/// <c>FloatVectorProperty</c> produces. The vector types are positional records, so the default
/// object-shaped binding cannot read that form and every vector parameter would fail to bind.
/// Arrays also keep the graph documents compact and match how the shader-side types are written.
/// </summary>
internal static class VectorJson
{
    internal static void ReadComponents(
        ref Utf8JsonReader reader,
        scoped Span<float> components,
        string typeName)
    {
        if (reader.TokenType != JsonTokenType.StartArray)
        {
            throw new JsonException($"{typeName} must be a JSON array of {components.Length} numbers.");
        }

        for (var index = 0; index < components.Length; index++)
        {
            if (!reader.Read() || reader.TokenType == JsonTokenType.EndArray)
            {
                throw new JsonException(
                    $"{typeName} requires {components.Length} components; the array is shorter.");
            }
            if (reader.TokenType != JsonTokenType.Number)
            {
                throw new JsonException($"{typeName} components must be numbers.");
            }
            components[index] = reader.GetSingle();
        }

        if (!reader.Read() || reader.TokenType != JsonTokenType.EndArray)
        {
            // Silently ignoring extra components would let a float4 bind into a float3 with one
            // value quietly discarded.
            throw new JsonException($"{typeName} requires exactly {components.Length} components.");
        }
    }

    internal static void WriteComponents(Utf8JsonWriter writer, ReadOnlySpan<float> components)
    {
        writer.WriteStartArray();
        foreach (var component in components)
        {
            writer.WriteNumberValue(component);
        }
        writer.WriteEndArray();
    }
}

internal sealed class Float2JsonConverter : JsonConverter<float2>
{
    public override float2 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        Span<float> components = stackalloc float[2];
        VectorJson.ReadComponents(ref reader, components, "float2");
        return new float2(components[0], components[1]);
    }

    public override void Write(Utf8JsonWriter writer, float2 value, JsonSerializerOptions options)
        => VectorJson.WriteComponents(writer, stackalloc float[] { value.X, value.Y });
}

internal sealed class Float3JsonConverter : JsonConverter<float3>
{
    public override float3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        Span<float> components = stackalloc float[3];
        VectorJson.ReadComponents(ref reader, components, "float3");
        return new float3(components[0], components[1], components[2]);
    }

    public override void Write(Utf8JsonWriter writer, float3 value, JsonSerializerOptions options)
        => VectorJson.WriteComponents(writer, stackalloc float[] { value.X, value.Y, value.Z });
}

internal sealed class Float4JsonConverter : JsonConverter<float4>
{
    public override float4 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        Span<float> components = stackalloc float[4];
        VectorJson.ReadComponents(ref reader, components, "float4");
        return new float4(components[0], components[1], components[2], components[3]);
    }

    public override void Write(Utf8JsonWriter writer, float4 value, JsonSerializerOptions options)
        => VectorJson.WriteComponents(
            writer,
            stackalloc float[] { value.X, value.Y, value.Z, value.W });
}
