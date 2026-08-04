namespace Feather.Blender.RenderHost;

/// <summary>
/// Retains the decoded snapshot and flattened CPU scene for the long-lived render-host process.
/// </summary>
internal sealed class RenderSceneCache
{
    private CachedSnapshot? snapshot;
    private string resourceFingerprint = string.Empty;
    private RenderSceneResources? resources;

    public CachedScene Resolve(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        path = Path.GetFullPath(path);
        var identity = SnapshotFileIdentity.Read(path);
        if (snapshot is { } current && current.Matches(path, identity))
        {
            return new CachedScene(
                current.GenerationId,
                current.ContentFingerprint,
                resources!,
                0.0,
                0.0);
        }

        var stage = System.Diagnostics.Stopwatch.StartNew();
        var loaded = SceneSnapshot.Load(path);
        stage.Stop();
        var loadMilliseconds = stage.Elapsed.TotalMilliseconds;
        var resourceHit = resources is not null &&
            string.Equals(resourceFingerprint, loaded.ContentFingerprint, StringComparison.Ordinal);
        var buildMilliseconds = 0.0;
        if (!resourceHit)
        {
            stage.Restart();
            resources = SceneResourceBuilder.Build(loaded);
            stage.Stop();
            buildMilliseconds = stage.Elapsed.TotalMilliseconds;
            resourceFingerprint = loaded.ContentFingerprint;
        }

        snapshot = new CachedSnapshot(
            path,
            identity,
            loaded.Metadata.GenerationId,
            loaded.ContentFingerprint);
        return new CachedScene(
            loaded.Metadata.GenerationId,
            loaded.ContentFingerprint,
            resources!,
            loadMilliseconds,
            buildMilliseconds);
    }

    private sealed record CachedSnapshot(
        string Path,
        SnapshotFileIdentity Identity,
        string GenerationId,
        string ContentFingerprint)
    {
        public bool Matches(string path, SnapshotFileIdentity identity)
            => string.Equals(Path, path, PathComparison) && Identity == identity;
    }

    private readonly record struct SnapshotFileIdentity(long Length, long LastWriteTicks)
    {
        public static SnapshotFileIdentity Read(string path)
        {
            var info = new FileInfo(path);
            if (!info.Exists)
            {
                throw new FileNotFoundException("Scene snapshot was not found.", path);
            }
            return new SnapshotFileIdentity(info.Length, info.LastWriteTimeUtc.Ticks);
        }
    }

    private static StringComparison PathComparison
        => OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
}

internal sealed record CachedScene(
    string GenerationId,
    string ContentFingerprint,
    RenderSceneResources Resources,
    double LoadMilliseconds,
    double BuildMilliseconds);

/// <summary>
/// Owns immutable resources whose lifetime is one pass-assembly generation. Keeping this separate
/// from graph-scoped pools lets warmed pipelines survive pass and View switches without allowing
/// graph textures or scene-derived buffers to outlive their identities.
/// </summary>
internal sealed class PassAssemblyResourcePool : IDisposable
{
    private readonly Dictionary<string, IDisposable> entries = new(StringComparer.Ordinal);
    private bool disposed;

    public T GetOrCreate<T>(string identity, Func<T> factory)
        where T : class, IDisposable
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (entries.TryGetValue(identity, out var existing))
        {
            if (existing is T typed)
            {
                return typed;
            }
            throw new InvalidOperationException(
                $"Retained assembly resource '{identity}' was requested with two different types.");
        }

        var created = factory() ?? throw new InvalidOperationException(
            $"Retained assembly resource factory '{identity}' returned null.");
        entries.Add(identity, created);
        return created;
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }
        foreach (var entry in entries.Values)
        {
            entry.Dispose();
        }
        entries.Clear();
        disposed = true;
    }
}

/// <summary>
/// Owns pass-created CPU and GPU scene resources for one pass-assembly generation.
/// </summary>
internal sealed class PassSceneResourcePool : IDisposable
{
    private readonly Dictionary<string, IDisposable> entries = new(StringComparer.Ordinal);
    private string sceneFingerprint = string.Empty;
    private string graphFingerprint = string.Empty;
    private bool disposed;

    public void Prepare(string nextSceneFingerprint, string nextGraphFingerprint)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (string.Equals(sceneFingerprint, nextSceneFingerprint, StringComparison.Ordinal) &&
            string.Equals(graphFingerprint, nextGraphFingerprint, StringComparison.Ordinal))
        {
            return;
        }

        ReleaseAll();
        sceneFingerprint = nextSceneFingerprint;
        graphFingerprint = nextGraphFingerprint;
    }

    public T GetOrCreate<T>(string identity, Func<T> factory)
        where T : class, IDisposable
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (entries.TryGetValue(identity, out var existing))
        {
            if (existing is T typed)
            {
                return typed;
            }
            throw new InvalidOperationException(
                $"Retained scene resource '{identity}' was requested with two different types.");
        }

        var created = factory() ?? throw new InvalidOperationException(
            $"Retained scene resource factory '{identity}' returned null.");
        entries.Add(identity, created);
        return created;
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }
        ReleaseAll();
        disposed = true;
    }

    private void ReleaseAll()
    {
        foreach (var entry in entries.Values)
        {
            entry.Dispose();
        }
        entries.Clear();
    }
}
