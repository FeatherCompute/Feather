namespace Feather.NN;

/// <summary>
/// Saves and loads Feather NN float parameters.
/// </summary>
/// <remarks>
/// Two on-disk versions exist. Version 1 is magic, version, count, then one entry per parameter.
/// Version 2 inserts a metadata block after the version field and before the count, so a host can read
/// provenance — step, loss, model kind, tags — without owning matching parameters. Both versions read;
/// <see cref="Save" /> still writes version 1 so a file written by this build stays readable by an older
/// one, and <see cref="SaveAtomic" /> writes version 2 when metadata is supplied.
/// </remarks>
public static class Checkpoint
{
    private const uint Magic = 0x46544843; // FTHC
    private const uint Version = 1;
    private const uint MetadataVersion = 2;

    /// <summary>
    /// Saves all float parameters to a checkpoint file.
    /// </summary>
    /// <remarks>
    /// Writes in place and is therefore not safe against a concurrent reader. Prefer
    /// <see cref="SaveAtomic" /> anywhere a renderer may be reading the same path.
    /// </remarks>
    /// <param name="path">The destination file path.</param>
    /// <param name="parameters">The parameters to write; non-float parameters are skipped.</param>
    public static void Save(string path, IEnumerable<IParameter> parameters)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(parameters);

        using var stream = File.Create(path);
        WriteTo(stream, parameters, metadata: null);
    }

    /// <summary>
    /// Writes to a temporary sibling file and atomically replaces the destination, so a concurrent
    /// reader never observes a partial checkpoint.
    /// </summary>
    /// <remarks>
    /// The same discipline the frame writer uses: write a temp file, close it, then move it over the
    /// destination. A reader either sees the previous complete file or the new complete file, never a
    /// half-written one. A crash mid-write leaves the temp file behind and the destination untouched.
    ///
    /// The temp file is a sibling rather than in the system temp directory, because a move across
    /// volumes is a copy and loses atomicity.
    /// </remarks>
    /// <param name="path">The destination file path.</param>
    /// <param name="parameters">The parameters to write; non-float parameters are skipped.</param>
    /// <param name="metadata">Optional provenance. Supplying it writes a version 2 checkpoint.</param>
    public static void SaveAtomic(string path, IEnumerable<IParameter> parameters, CheckpointMetadata? metadata = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(parameters);

        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var temporaryPath = $"{fullPath}.{Environment.ProcessId:x}{DateTime.UtcNow.Ticks:x}.tmp";
        try
        {
            using (var stream = File.Create(temporaryPath))
            {
                WriteTo(stream, parameters, metadata);
                stream.Flush(flushToDisk: true);
            }

            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        catch
        {
            TryDelete(temporaryPath);
            throw;
        }
    }

    /// <summary>
    /// Loads parameter values from a checkpoint file into matching named parameters.
    /// </summary>
    /// <remarks>
    /// Names present in the file but absent from <paramref name="parameters" /> are skipped silently.
    /// That is the historical behavior and is preserved; <see cref="LoadStrict" /> is the version that
    /// reports instead, and is what a renamed layer needs to avoid loading a plausible-looking partial
    /// model.
    /// </remarks>
    /// <param name="path">The checkpoint file path.</param>
    /// <param name="parameters">The parameters to fill.</param>
    public static void Load(string path, IEnumerable<IParameter> parameters)
        => LoadCore(path, parameters, strict: false);

    /// <summary>
    /// Loads and reports what happened instead of silently skipping unmatched names.
    /// </summary>
    /// <remarks>
    /// Fails loudly on a truncated file, an unknown version, or a shape mismatch. Name mismatches are
    /// reported rather than thrown, because a caller loading a subset of a larger checkpoint is a
    /// legitimate case — it is silence about them that is not. Inspect
    /// <see cref="CheckpointLoadResult.IsComplete" /> when a full match is required, or call
    /// <see cref="CheckpointLoadResult.EnsureComplete" /> to turn a partial match into a throw.
    ///
    /// Every unreadable-file failure is an <see cref="InvalidDataException" />: wrong magic, unknown
    /// version, negative counts, truncation, and trailing bytes alike. One exception type means a host
    /// showing "this checkpoint is unusable" needs one catch clause.
    /// </remarks>
    /// <param name="path">The checkpoint file path.</param>
    /// <param name="parameters">The parameters to fill.</param>
    /// <returns>Which names loaded, which were missing from the file, and which the file did not use.</returns>
    /// <exception cref="InvalidDataException">The file is not a readable checkpoint.</exception>
    public static CheckpointLoadResult LoadStrict(string path, IEnumerable<IParameter> parameters)
        => LoadCore(path, parameters, strict: true);

    /// <summary>
    /// Reads the header and parameter table without touching the GPU.
    /// </summary>
    /// <remarks>
    /// This is what lets a UI show "step 4200, loss 0.0031, 6 tensors" for a checkpoint it has no
    /// matching parameters for. Values are skipped rather than read, so the cost is proportional to the
    /// entry count and not to the weight count. A future version number fails with a clear message
    /// rather than misreading the file.
    /// </remarks>
    /// <param name="path">The checkpoint file path.</param>
    /// <exception cref="InvalidDataException">The file is not a readable checkpoint.</exception>
    public static CheckpointInfo Inspect(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        var header = ReadHeader(reader, path);
        var entries = new List<CheckpointEntryInfo>(header.Count);
        for (var i = 0; i < header.Count; i++)
        {
            var entry = ReadEntryHeader(reader);
            entries.Add(new CheckpointEntryInfo(entry.Name, new TensorShape(entry.Dimensions), entry.ValueCount));
            SkipValues(reader, entry.ValueCount);
        }

        return new CheckpointInfo(header.Version, header.Metadata, entries);
    }

    /// <summary>Reads a checkpoint's file stamp, or null when the file does not exist.</summary>
    /// <param name="path">The checkpoint file path.</param>
    public static CheckpointStamp? TryReadStamp(string path)
        => CheckpointStamp.TryRead(path);

    private static void WriteTo(Stream stream, IEnumerable<IParameter> parameters, CheckpointMetadata? metadata)
    {
        var floatParameters = ParameterValidation.EnsureUnique(parameters, nameof(parameters))
            .OfType<Parameter<float>>()
            .ToArray();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(metadata is null ? Version : MetadataVersion);
        if (metadata is not null)
        {
            WriteMetadata(writer, metadata);
        }

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

    private static CheckpointLoadResult LoadCore(string path, IEnumerable<IParameter> parameters, bool strict)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(parameters);

        var floatParameters = ParameterValidation.EnsureUnique(parameters, nameof(parameters))
            .OfType<Parameter<float>>()
            .ToDictionary(parameter => parameter.FullName, StringComparer.Ordinal);
        var loaded = new List<string>();
        var unusedInFile = new List<string>();

        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        var header = ReadHeader(reader, path);
        for (var i = 0; i < header.Count; i++)
        {
            var entry = ReadEntryHeader(reader);
            var values = ReadValues(reader, entry.ValueCount, entry.Name);
            if (!floatParameters.TryGetValue(entry.Name, out var parameter))
            {
                unusedInFile.Add(entry.Name);
                continue;
            }

            if (!parameter.Value.Shape.Equals(new TensorShape(entry.Dimensions)))
            {
                throw new InvalidDataException($"Checkpoint parameter '{entry.Name}' shape does not match the target parameter.");
            }

            // Upload only after shape validation so partially applied checkpoints are avoided for mismatched entries.
            parameter.Value.Buffer.Upload(values);
            loaded.Add(entry.Name);
        }

        if (strict && stream.Position != stream.Length)
        {
            throw new InvalidDataException($"Checkpoint '{path}' has {stream.Length - stream.Position} trailing bytes after {header.Count} entries.");
        }

        var missingFromFile = floatParameters.Keys
            .Where(name => !loaded.Contains(name, StringComparer.Ordinal))
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();
        return new CheckpointLoadResult(header.Version, header.Metadata, loaded, missingFromFile, unusedInFile);
    }

    private static CheckpointHeader ReadHeader(BinaryReader reader, string path)
    {
        if (ReadUInt32(reader) != Magic)
        {
            throw new InvalidDataException($"'{path}' is not a Feather checkpoint.");
        }

        var version = ReadUInt32(reader);
        var metadata = version switch
        {
            Version => null,
            MetadataVersion => ReadMetadata(reader),
            _ => throw new InvalidDataException($"'{path}' is a version {version} Feather checkpoint, but this build reads versions {Version} and {MetadataVersion}.")
        };

        var count = ReadInt32(reader);
        if (count < 0)
        {
            throw new InvalidDataException($"Checkpoint '{path}' declares {count} entries.");
        }

        return new CheckpointHeader(version, metadata, count);
    }

    private static void WriteMetadata(BinaryWriter writer, CheckpointMetadata metadata)
    {
        writer.Write(metadata.Step);
        writer.Write(metadata.Loss);
        writer.Write(metadata.SavedAtUtcTicks);
        WriteString(writer, metadata.ModelKind ?? string.Empty);
        var tags = metadata.Tags ?? new Dictionary<string, string>(StringComparer.Ordinal);
        writer.Write(tags.Count);
        foreach (var tag in tags)
        {
            WriteString(writer, tag.Key);
            WriteString(writer, tag.Value);
        }
    }

    private static CheckpointMetadata ReadMetadata(BinaryReader reader)
    {
        var step = ReadInt32(reader);
        var loss = ReadSingle(reader);
        var savedAtUtcTicks = ReadInt64(reader);
        var modelKind = ReadString(reader);
        var tagCount = ReadInt32(reader);
        if (tagCount < 0)
        {
            throw new InvalidDataException($"Checkpoint metadata declares {tagCount} tags.");
        }

        var tags = new Dictionary<string, string>(tagCount, StringComparer.Ordinal);
        for (var i = 0; i < tagCount; i++)
        {
            var key = ReadString(reader);
            tags[key] = ReadString(reader);
        }

        return new CheckpointMetadata(step, loss, string.IsNullOrEmpty(modelKind) ? null : modelKind, tags)
        {
            SavedAtUtcTicks = savedAtUtcTicks
        };
    }

    private static CheckpointEntryHeader ReadEntryHeader(BinaryReader reader)
    {
        var name = ReadString(reader);
        var rank = ReadInt32(reader);
        if (rank < 0)
        {
            throw new InvalidDataException($"Checkpoint entry '{name}' declares rank {rank}.");
        }

        var dimensions = new int[rank];
        for (var d = 0; d < dimensions.Length; d++)
        {
            dimensions[d] = ReadInt32(reader);
        }

        var valueCount = ReadInt32(reader);
        if (valueCount < 0)
        {
            throw new InvalidDataException($"Checkpoint entry '{name}' declares {valueCount} values.");
        }

        return new CheckpointEntryHeader(name, dimensions, valueCount);
    }

    private static float[] ReadValues(BinaryReader reader, int valueCount, string name)
    {
        // Bounds-checked up front so a truncated entry names itself, rather than surfacing as a bare
        // EndOfStreamException from somewhere inside the loop.
        var byteCount = (long)valueCount * sizeof(float);
        var stream = reader.BaseStream;
        if (stream.CanSeek && stream.Position + byteCount > stream.Length)
        {
            throw new InvalidDataException($"Checkpoint entry '{name}' declares {valueCount} values but only {(stream.Length - stream.Position) / sizeof(float)} remain in the file.");
        }

        var values = new float[valueCount];
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = reader.ReadSingle();
        }

        return values;
    }

    private static void SkipValues(BinaryReader reader, int valueCount)
    {
        var byteCount = (long)valueCount * sizeof(float);
        var stream = reader.BaseStream;
        if (stream.CanSeek)
        {
            if (stream.Position + byteCount > stream.Length)
            {
                throw new InvalidDataException($"Checkpoint declares {valueCount} values but only {stream.Length - stream.Position} bytes remain.");
            }

            stream.Seek(byteCount, SeekOrigin.Current);
            return;
        }

        var skipped = reader.ReadBytes(checked((int)byteCount));
        if (skipped.Length != byteCount)
        {
            throw new InvalidDataException($"Checkpoint declares {valueCount} values but the file ended early.");
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
            // A leftover temp file is preferable to masking the original failure.
        }
        catch (UnauthorizedAccessException)
        {
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
        var length = ReadInt32(reader);
        if (length < 0)
        {
            throw new InvalidDataException("Checkpoint string length is invalid.");
        }

        var bytes = reader.ReadBytes(length);
        if (bytes.Length != length)
        {
            throw new InvalidDataException("Checkpoint ended while reading a string.");
        }

        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    // BinaryReader throws EndOfStreamException for a truncated primitive, but with no indication of
    // what was being read. These keep a truncated header from surfacing as a bare stack trace, and
    // report it as InvalidDataException so a caller has one exception type covering every kind of bad
    // file rather than catching truncation separately from corruption.
    private static uint ReadUInt32(BinaryReader reader)
        => reader.BaseStream.Position + sizeof(uint) <= Length(reader)
            ? reader.ReadUInt32()
            : throw new InvalidDataException("Checkpoint ended inside its header.");

    private static int ReadInt32(BinaryReader reader)
        => reader.BaseStream.Position + sizeof(int) <= Length(reader)
            ? reader.ReadInt32()
            : throw new InvalidDataException("Checkpoint ended while reading an integer field.");

    private static long ReadInt64(BinaryReader reader)
        => reader.BaseStream.Position + sizeof(long) <= Length(reader)
            ? reader.ReadInt64()
            : throw new InvalidDataException("Checkpoint ended while reading a timestamp.");

    private static float ReadSingle(BinaryReader reader)
        => reader.BaseStream.Position + sizeof(float) <= Length(reader)
            ? reader.ReadSingle()
            : throw new InvalidDataException("Checkpoint ended while reading a float field.");

    private static long Length(BinaryReader reader)
        => reader.BaseStream.CanSeek ? reader.BaseStream.Length : long.MaxValue;

    private readonly record struct CheckpointHeader(uint Version, CheckpointMetadata? Metadata, int Count);

    private readonly record struct CheckpointEntryHeader(string Name, int[] Dimensions, int ValueCount);
}

/// <summary>
/// Optional provenance written into a version 2 checkpoint.
/// </summary>
/// <param name="Step">The training step the checkpoint was written at.</param>
/// <param name="Loss">The loss observed at that step, or NaN when it was not measured.</param>
/// <param name="ModelKind">An optional discriminator a host can display or validate against.</param>
/// <param name="Tags">Optional free-form key/value provenance.</param>
public sealed record CheckpointMetadata(
    int Step,
    float Loss,
    string? ModelKind = null,
    IReadOnlyDictionary<string, string>? Tags = null)
{
    /// <summary>Gets or sets when the checkpoint was written, in UTC ticks.</summary>
    /// <remarks>
    /// Defaults to construction time so a caller gets a usable timestamp without passing one, and is
    /// settable so a round-trip through the file preserves the original rather than stamping the read.
    /// </remarks>
    public long SavedAtUtcTicks { get; init; } = DateTime.UtcNow.Ticks;

    /// <summary>Gets when the checkpoint was written.</summary>
    public DateTime SavedAtUtc => new(SavedAtUtcTicks, DateTimeKind.Utc);
}

/// <summary>
/// A checkpoint's header and parameter table, read without loading weights onto the GPU.
/// </summary>
/// <param name="Version">The on-disk format version.</param>
/// <param name="Metadata">The provenance block, or null for a version 1 checkpoint.</param>
/// <param name="Entries">One entry per stored tensor.</param>
public sealed record CheckpointInfo(
    uint Version,
    CheckpointMetadata? Metadata,
    IReadOnlyList<CheckpointEntryInfo> Entries)
{
    /// <summary>Gets the total number of float values across every entry.</summary>
    public long WeightCount => Entries.Sum(entry => (long)entry.ValueCount);
}

/// <summary>
/// One tensor's identity and shape inside a checkpoint.
/// </summary>
/// <param name="FullName">The fully-qualified parameter name.</param>
/// <param name="Shape">The stored tensor shape.</param>
/// <param name="ValueCount">The number of float values stored for this entry.</param>
public sealed record CheckpointEntryInfo(string FullName, TensorShape Shape, int ValueCount);

/// <summary>
/// What a strict load matched, and what it did not.
/// </summary>
/// <param name="Version">The on-disk format version that was read.</param>
/// <param name="Metadata">The provenance block, or null for a version 1 checkpoint.</param>
/// <param name="Loaded">Names present in both the file and the supplied parameters.</param>
/// <param name="MissingFromFile">Supplied parameters the file did not contain.</param>
/// <param name="UnusedInFile">Names in the file that no supplied parameter matched.</param>
public sealed record CheckpointLoadResult(
    uint Version,
    CheckpointMetadata? Metadata,
    IReadOnlyList<string> Loaded,
    IReadOnlyList<string> MissingFromFile,
    IReadOnlyList<string> UnusedInFile)
{
    /// <summary>Gets a value indicating whether every supplied parameter and every file entry matched.</summary>
    public bool IsComplete => MissingFromFile.Count == 0 && UnusedInFile.Count == 0;

    /// <summary>Throws when the load was not an exact match, naming what did not line up.</summary>
    /// <remarks>
    /// Separate from <see cref="Checkpoint.LoadStrict" /> so a caller loading a deliberate subset is not
    /// forced to catch, while a caller expecting an exact model can demand one in a single line.
    /// </remarks>
    public CheckpointLoadResult EnsureComplete()
    {
        if (IsComplete)
        {
            return this;
        }

        var problems = new List<string>();
        if (MissingFromFile.Count > 0)
        {
            problems.Add($"missing from file: {string.Join(", ", MissingFromFile)}");
        }

        if (UnusedInFile.Count > 0)
        {
            problems.Add($"unused in file: {string.Join(", ", UnusedInFile)}");
        }

        throw new InvalidDataException($"Checkpoint did not match the supplied parameters ({string.Join("; ", problems)}).");
    }
}

/// <summary>
/// Length plus last-write time, matching the host's file-signature idiom.
/// </summary>
/// <remarks>
/// Deliberately not a content hash. The point is a cheap check a pass can run every iteration to decide
/// whether to reload; hashing the file would cost as much as reloading it.
/// </remarks>
/// <param name="Length">The file length in bytes.</param>
/// <param name="LastWriteTicks">The last-write timestamp in UTC ticks.</param>
public readonly record struct CheckpointStamp(long Length, long LastWriteTicks)
{
    /// <summary>Reads a file's stamp, or returns null when the file does not exist.</summary>
    /// <param name="path">The file path to stamp.</param>
    public static CheckpointStamp? TryRead(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var info = new FileInfo(path);
        return info.Exists ? new CheckpointStamp(info.Length, info.LastWriteTimeUtc.Ticks) : null;
    }
}
