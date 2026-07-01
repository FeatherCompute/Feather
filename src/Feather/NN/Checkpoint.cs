namespace Feather.NN;

/// <summary>
/// Saves and loads Feather NN float parameters.
/// </summary>
public static class Checkpoint
{
    private const uint Magic = 0x46544843; // FTHC
    private const uint Version = 1;

    /// <summary>
    /// Saves all float parameters to a checkpoint file.
    /// </summary>
    public static void Save(string path, IEnumerable<IParameter> parameters)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(parameters);

        var floatParameters = ParameterValidation.EnsureUnique(parameters, nameof(parameters))
            .OfType<Parameter<float>>()
            .ToArray();
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(Magic);
        writer.Write(Version);
        writer.Write(floatParameters.Length);

        foreach (var parameter in floatParameters)
        {
            WriteString(writer, parameter.FullName);
            writer.Write(parameter.Value.Shape.Rank);
            foreach (var dimension in parameter.Value.Shape.Dimensions)
            {
                writer.Write(dimension);
            }

            var values = parameter.Value.Buffer.ToArray();
            writer.Write(values.Length);
            foreach (var value in values)
            {
                writer.Write(value);
            }
        }
    }

    /// <summary>
    /// Loads parameter values from a checkpoint file into matching named parameters.
    /// </summary>
    public static void Load(string path, IEnumerable<IParameter> parameters)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(parameters);

        var floatParameters = ParameterValidation.EnsureUnique(parameters, nameof(parameters))
            .OfType<Parameter<float>>()
            .ToDictionary(parameter => parameter.FullName, StringComparer.Ordinal);
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        if (reader.ReadUInt32() != Magic || reader.ReadUInt32() != Version)
        {
            throw new InvalidDataException("The file is not a supported Feather checkpoint.");
        }

        var count = reader.ReadInt32();
        for (var i = 0; i < count; i++)
        {
            var name = ReadString(reader);
            var rank = reader.ReadInt32();
            var dimensions = new int[rank];
            for (var d = 0; d < dimensions.Length; d++)
            {
                dimensions[d] = reader.ReadInt32();
            }

            var valueCount = reader.ReadInt32();
            var values = new float[valueCount];
            for (var valueIndex = 0; valueIndex < values.Length; valueIndex++)
            {
                values[valueIndex] = reader.ReadSingle();
            }

            if (!floatParameters.TryGetValue(name, out var parameter))
            {
                continue;
            }

            if (!parameter.Value.Shape.Equals(new TensorShape(dimensions)))
            {
                throw new InvalidDataException($"Checkpoint parameter '{name}' shape does not match the target parameter.");
            }

            // Upload only after shape validation so partially applied checkpoints are avoided for mismatched entries.
            parameter.Value.Buffer.Upload(values);
        }
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(value);
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }

    private static string ReadString(BinaryReader reader)
    {
        var length = reader.ReadInt32();
        if (length < 0)
        {
            throw new InvalidDataException("Checkpoint string length is invalid.");
        }

        var bytes = reader.ReadBytes(length);
        if (bytes.Length != length)
        {
            throw new EndOfStreamException("Checkpoint ended while reading a string.");
        }

        return System.Text.Encoding.UTF8.GetString(bytes);
    }
}
