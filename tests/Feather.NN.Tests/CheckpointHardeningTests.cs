using Feather.NN;

namespace Feather.NN.Tests;

/// <summary>
/// Covers the checkpoint guarantees a long-lived host depends on: an atomic replace that a crash cannot
/// half-apply, a strict load that reports rather than silently skips, metadata readable without owning
/// matching parameters, and backward compatibility with version 1 files.
/// </summary>
public class CheckpointHardeningTests : IDisposable
{
    private readonly string directory = Path.Combine(Path.GetTempPath(), $"feather-ckpt-{Guid.NewGuid():N}");

    public CheckpointHardeningTests() => Directory.CreateDirectory(directory);

    public void Dispose()
    {
        if (Directory.Exists(directory))
        {
            Directory.Delete(directory, recursive: true);
        }

        GC.SuppressFinalize(this);
    }

    private string Path_(string name) => Path.Combine(directory, name);

    private static Parameter<float> FloatParameter(string name, params float[] values)
        => new(
            name,
            new Tensor<float>(new TensorShape(values.Length), GPU.CreateBuffer<float>(values)),
            new Tensor<float>(new TensorShape(values.Length), GPU.CreateBuffer<float>(values.Length)));

    [Fact]
    public void SaveAtomicRoundTripsValuesAndMetadata()
    {
        var path = Path_("round-trip.fthc");
        using var source = FloatParameter("w", 1f, 2f, 3f, 4f);
        using var target = FloatParameter("w", 0f, 0f, 0f, 0f);
        var metadata = new CheckpointMetadata(
            4200,
            0.0031f,
            ModelKind: "mlp-regression-3to1",
            Tags: new Dictionary<string, string> { ["hiddenSize"] = "12" });

        Checkpoint.SaveAtomic(path, [source], metadata);
        var result = Checkpoint.LoadStrict(path, [target]);

        Assert.Equal([1f, 2f, 3f, 4f], target.Value.Buffer.ToArray());
        Assert.True(result.IsComplete);
        Assert.Equal(2u, result.Version);
        Assert.NotNull(result.Metadata);
        Assert.Equal(4200, result.Metadata!.Step);
        Assert.Equal(0.0031f, result.Metadata.Loss, 6);
        Assert.Equal("mlp-regression-3to1", result.Metadata.ModelKind);
        Assert.Equal("12", result.Metadata.Tags!["hiddenSize"]);

        // The timestamp must survive the file rather than being restamped on read.
        Assert.Equal(metadata.SavedAtUtcTicks, result.Metadata.SavedAtUtcTicks);
    }

    [Fact]
    public void SaveAtomicLeavesNoTemporaryFilesBehind()
    {
        var path = Path_("no-temps.fthc");
        using var parameter = FloatParameter("w", 7f);

        Checkpoint.SaveAtomic(path, [parameter]);
        Checkpoint.SaveAtomic(path, [parameter]);

        var remaining = Directory.GetFiles(directory).Select(file => Path.GetFileName(file) ?? string.Empty).ToArray();
        Assert.Equal([Path.GetFileName(path)], remaining);
    }

    [Fact]
    public void SaveAtomicWritesVersionOneWhenNoMetadataIsSupplied()
    {
        var path = Path_("no-metadata.fthc");
        using var parameter = FloatParameter("w", 1f);

        Checkpoint.SaveAtomic(path, [parameter]);

        var info = Checkpoint.Inspect(path);
        Assert.Equal(1u, info.Version);
        Assert.Null(info.Metadata);
    }

    [Fact]
    public void SaveKeepsWritingVersionOneForBackwardCompatibility()
    {
        var path = Path_("legacy.fthc");
        using var source = FloatParameter("w", 5f, 6f);
        using var target = FloatParameter("w", 0f, 0f);

        Checkpoint.Save(path, [source]);

        var info = Checkpoint.Inspect(path);
        Assert.Equal(1u, info.Version);

        // A version 1 file must still load through both the old and the new entry point.
        Checkpoint.Load(path, [target]);
        Assert.Equal([5f, 6f], target.Value.Buffer.ToArray());

        var strict = Checkpoint.LoadStrict(path, [target]);
        Assert.True(strict.IsComplete);
        Assert.Null(strict.Metadata);
    }

    [Fact]
    public void AtomicReplaceLeavesThePreviousCheckpointIntactWhenTheWriteIsInterrupted()
    {
        // The crash simulation: SaveAtomic's temp-then-move is reproduced with the move omitted, which is
        // exactly the state a process killed mid-save leaves behind. The destination must still hold the
        // old, complete checkpoint.
        var path = Path_("interrupted.fthc");
        using var oldValues = FloatParameter("w", 1f, 1f, 1f);
        using var newValues = FloatParameter("w", 9f, 9f, 9f);
        Checkpoint.SaveAtomic(path, [oldValues]);
        var stampBefore = Checkpoint.TryReadStamp(path);

        var temporaryPath = $"{path}.interrupted.tmp";
        Checkpoint.Save(temporaryPath, [newValues]);

        using var target = FloatParameter("w", 0f, 0f, 0f);
        Checkpoint.LoadStrict(path, [target]).EnsureComplete();

        Assert.Equal([1f, 1f, 1f], target.Value.Buffer.ToArray());
        Assert.Equal(stampBefore, Checkpoint.TryReadStamp(path));

        // And once the move lands, the new values are what a reader sees. No intermediate state exists
        // between those two observations.
        File.Move(temporaryPath, path, overwrite: true);
        Checkpoint.LoadStrict(path, [target]).EnsureComplete();
        Assert.Equal([9f, 9f, 9f], target.Value.Buffer.ToArray());
    }

    [Fact]
    public void LoadStrictThrowsOnATruncatedFile()
    {
        var path = Path_("truncated.fthc");
        using var source = FloatParameter("w", 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
        Checkpoint.SaveAtomic(path, [source]);

        var bytes = File.ReadAllBytes(path);
        File.WriteAllBytes(path, bytes[..(bytes.Length - 12)]);

        using var target = FloatParameter("w", 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f);
        var ex = Assert.Throws<InvalidDataException>(() => Checkpoint.LoadStrict(path, [target]));
        Assert.Contains("w", ex.Message);
    }

    [Fact]
    public void LoadStrictThrowsOnAnUnknownVersion()
    {
        var path = Path_("future.fthc");
        using var source = FloatParameter("w", 1f);
        Checkpoint.SaveAtomic(path, [source]);

        var bytes = File.ReadAllBytes(path);
        BitConverter.GetBytes(99u).CopyTo(bytes, 4);
        File.WriteAllBytes(path, bytes);

        using var target = FloatParameter("w", 0f);
        var ex = Assert.Throws<InvalidDataException>(() => Checkpoint.LoadStrict(path, [target]));
        Assert.Contains("version 99", ex.Message);
    }

    [Fact]
    public void LoadStrictThrowsOnANonCheckpointFile()
    {
        var path = Path_("garbage.fthc");
        File.WriteAllBytes(path, [1, 2, 3, 4, 5, 6, 7, 8]);

        using var target = FloatParameter("w", 0f);
        var ex = Assert.Throws<InvalidDataException>(() => Checkpoint.LoadStrict(path, [target]));
        Assert.Contains("not a Feather checkpoint", ex.Message);
    }

    [Fact]
    public void LoadStrictReportsRenamedParametersInsteadOfSilentlySkippingThem()
    {
        var path = Path_("renamed.fthc");
        using var source = FloatParameter("hidden.weight", 1f, 2f);
        Checkpoint.SaveAtomic(path, [source]);

        using var target = FloatParameter("hidden.w", 0f, 0f);
        var result = Checkpoint.LoadStrict(path, [target]);

        Assert.False(result.IsComplete);
        Assert.Equal(["hidden.w"], result.MissingFromFile);
        Assert.Equal(["hidden.weight"], result.UnusedInFile);
        Assert.Empty(result.Loaded);

        var ex = Assert.Throws<InvalidDataException>(result.EnsureComplete);
        Assert.Contains("hidden.w", ex.Message);

        // Load's historical silent-skip behavior is preserved for callers that depend on it.
        Checkpoint.Load(path, [target]);
        Assert.Equal([0f, 0f], target.Value.Buffer.ToArray());
    }

    [Fact]
    public void LoadStrictThrowsOnAShapeMismatch()
    {
        var path = Path_("shape.fthc");
        using var source = FloatParameter("w", 1f, 2f, 3f);
        Checkpoint.SaveAtomic(path, [source]);

        using var target = FloatParameter("w", 0f, 0f);
        var ex = Assert.Throws<InvalidDataException>(() => Checkpoint.LoadStrict(path, [target]));
        Assert.Contains("shape does not match", ex.Message);

        // The mismatched entry must not have been partially uploaded.
        Assert.Equal([0f, 0f], target.Value.Buffer.ToArray());
    }

    [Fact]
    public void InspectReadsMetadataAndShapesWithoutMatchingParameters()
    {
        var path = Path_("inspect.fthc");
        using var first = FloatParameter("a.weight", 1f, 2f, 3f, 4f);
        using var second = FloatParameter("b.bias", 5f);
        Checkpoint.SaveAtomic(path, [first, second], new CheckpointMetadata(17, 0.5f, ModelKind: "probe"));

        var info = Checkpoint.Inspect(path);

        Assert.Equal(2u, info.Version);
        Assert.Equal(17, info.Metadata!.Step);
        Assert.Equal("probe", info.Metadata.ModelKind);
        Assert.Equal(5L, info.WeightCount);
        Assert.Equal(["a.weight", "b.bias"], info.Entries.Select(entry => entry.FullName).ToArray());
        Assert.Equal(4, info.Entries[0].ValueCount);
        Assert.Equal(new TensorShape(4), info.Entries[0].Shape);
    }

    [Fact]
    public void StampChangesWhenTheCheckpointIsRewritten()
    {
        var path = Path_("stamp.fthc");
        using var small = FloatParameter("w", 1f);
        using var large = FloatParameter("w", 1f, 2f, 3f, 4f);

        Assert.Null(Checkpoint.TryReadStamp(path));

        Checkpoint.SaveAtomic(path, [small]);
        var first = Checkpoint.TryReadStamp(path);
        Assert.NotNull(first);

        Checkpoint.SaveAtomic(path, [large]);
        var second = Checkpoint.TryReadStamp(path);

        Assert.NotEqual(first, second);
    }
}
