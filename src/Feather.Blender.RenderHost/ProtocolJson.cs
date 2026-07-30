using System.Text.Json;

namespace Feather.Blender.RenderHost;

internal static class ProtocolJson
{
    public static JsonSerializerOptions Options { get; } = new(JsonSerializerDefaults.Web)
    {
        PropertyNameCaseInsensitive = false,
        WriteIndented = false,
        // Vector parameters arrive as JSON arrays. The vector types are positional records, so
        // without these converters every vector pass parameter fails to bind.
        Converters =
        {
            new Float2JsonConverter(),
            new Float3JsonConverter(),
            new Float4JsonConverter()
        }
    };
}
