using Feather.Resources;

namespace Feather.NN;

/// <summary>
/// A read-only, GPU-resident set of named weight tensors loaded from a checkpoint, intended for
/// binding into inference kernels and shaders.
/// </summary>
/// <remarks>
/// The interop primitive already existed — <see cref="Tensor{T}.AsReadOnlyBuffer" /> hands a shader a
/// buffer view directly, so nothing here converts anything. What this type adds is lifecycle: a
/// checkpoint's tensors loaded once, kept alive as a unit, and stamped so a caller can tell whether the
/// file changed underneath it.
///
/// Lifetime rules, because getting them wrong is the failure mode this type exists to prevent:
///
/// <list type="bullet">
/// <item>The buffer from <see cref="Buffer" /> is a view, not a copy. It is valid only while this
/// instance is alive; disposing the weights invalidates every view taken from them.</item>
/// <item>Whoever loaded the weights disposes them. Code that received an instance from a cache — a
/// render pass calling into <see cref="InferenceWeightsCache" />, for example — must not dispose it,
/// because the next iteration expects it to still be there.</item>
/// <item>Do not load inside a per-frame code path. A host recreates a pass per execution, so a
/// <see cref="Load" /> in <c>Execute</c> re-reads the file and re-uploads every weight every frame. Go
/// through a cache instead.</item>
/// </list>
///
/// Loading uses <see cref="Checkpoint.LoadStrict" />, so a renamed layer is reported rather than
/// yielding a partially-populated model that renders plausible garbage.
/// </remarks>
public sealed class InferenceWeights : IDisposable
{
    private readonly Dictionary<string, Tensor<float>> tensors;
    private bool disposed;

    private InferenceWeights(
        string sourcePath,
        CheckpointStamp stamp,
        CheckpointMetadata? metadata,
        Dictionary<string, Tensor<float>> tensors)
    {
        SourcePath = sourcePath;
        Stamp = stamp;
        Metadata = metadata;
        this.tensors = tensors;
    }

    /// <summary>
    /// Loads every float tensor in the checkpoint by name.
    /// </summary>
    /// <remarks>
    /// Allocates one GPU buffer per stored tensor and uploads its values. The stamp is read before the
    /// contents, so a file rewritten during the load is detected as stale on the next check rather than
    /// being cached as current.
    /// </remarks>
    /// <param name="path">The checkpoint file path.</param>
    public static InferenceWeights Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var fullPath = Path.GetFullPath(path);
        var stamp = CheckpointStamp.TryRead(fullPath)
            ?? throw new FileNotFoundException($"No checkpoint exists at '{fullPath}'.", fullPath);

        var info = Checkpoint.Inspect(fullPath);
        var tensors = new Dictionary<string, Tensor<float>>(info.Entries.Count, StringComparer.Ordinal);
        var parameters = new List<Parameter<float>>(info.Entries.Count);
        try
        {
            foreach (var entry in info.Entries)
            {
                // Parameter<float> requires a gradient tensor shaped like its value, so loading costs a
                // transient second buffer per tensor. It is freed as soon as the load finishes; only the
                // values outlive this method.
                var value = new Tensor<float>(entry.Shape, GPU.CreateBuffer<float>(entry.Shape.ElementCount));
                var gradient = new Tensor<float>(entry.Shape, GPU.CreateBuffer<float>(entry.Shape.ElementCount));
                var parameter = new Parameter<float>(entry.FullName, value, gradient);
                if (!string.Equals(parameter.FullName, entry.FullName, StringComparison.Ordinal))
                {
                    throw new InvalidDataException($"Checkpoint entry '{entry.FullName}' could not be represented as a parameter name.");
                }

                parameters.Add(parameter);
                tensors.Add(entry.FullName, value);
            }

            Checkpoint.LoadStrict(fullPath, parameters).EnsureComplete();

            // The gradient tensors are scaffolding for the load; only the values outlive it.
            foreach (var parameter in parameters)
            {
                parameter.Gradient.Dispose();
            }

            return new InferenceWeights(fullPath, stamp, info.Metadata, tensors);
        }
        catch
        {
            foreach (var parameter in parameters)
            {
                parameter.Gradient.Dispose();
            }

            foreach (var tensor in tensors.Values)
            {
                tensor.Dispose();
            }

            throw;
        }
    }

    /// <summary>Gets the absolute path the weights were loaded from.</summary>
    public string SourcePath { get; }

    /// <summary>Gets the checkpoint's file identity, for cache invalidation.</summary>
    public CheckpointStamp Stamp { get; }

    /// <summary>Gets the checkpoint's provenance, or null for a version 1 checkpoint.</summary>
    public CheckpointMetadata? Metadata { get; }

    /// <summary>Gets the stored tensor names.</summary>
    public IReadOnlyCollection<string> Names
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return tensors.Keys;
        }
    }

    /// <summary>Gets a stored tensor by its fully-qualified name.</summary>
    /// <param name="fullName">The fully-qualified parameter name.</param>
    public Tensor<float> this[string fullName]
        => TryGet(fullName, out var tensor)
            ? tensor
            : throw new KeyNotFoundException($"'{SourcePath}' contains no tensor named '{fullName}'. Available: {string.Join(", ", tensors.Keys)}.");

    /// <summary>Attempts to read a stored tensor by name.</summary>
    /// <param name="fullName">The fully-qualified parameter name.</param>
    /// <param name="tensor">The stored tensor when present.</param>
    public bool TryGet(string fullName, out Tensor<float> tensor)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(fullName);
        return tensors.TryGetValue(fullName, out tensor!);
    }

    /// <summary>
    /// Gets a shader-facing view for binding into a generated kernel or shader.
    /// </summary>
    /// <remarks>
    /// A view over GPU memory this instance owns, not a copy. Bind it and dispatch; do not keep it past
    /// the lifetime of these weights. See the type's remarks for the full lifecycle rules.
    /// </remarks>
    /// <param name="fullName">The fully-qualified parameter name.</param>
    public ReadOnlyBuffer<float> Buffer(string fullName)
        => this[fullName].AsReadOnlyBuffer();

    /// <summary>
    /// Gets a value indicating whether the source file still matches the stamp these weights were
    /// loaded under.
    /// </summary>
    /// <remarks>
    /// A deleted file counts as stale. Cheap enough to call every iteration — it is one file stat.
    /// </remarks>
    public bool IsStale()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        var current = CheckpointStamp.TryRead(SourcePath);
        return current is null || !current.Value.Equals(Stamp);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        foreach (var tensor in tensors.Values)
        {
            tensor.Dispose();
        }

        tensors.Clear();
        disposed = true;
    }
}

/// <summary>
/// A host-owned cache of checkpoint weights, reused across iterations and pass instances.
/// </summary>
/// <remarks>
/// This exists because of where pass lifetime sits: a host creates a fresh pass instance per execution,
/// so a pass cannot hold weights across iterations no matter how it is written. The cache must be owned
/// one level up, alongside the texture and raster-target pools, which is the same placement that makes
/// GPU-resident history survive.
///
/// A host wires it to its render context — <c>context.GetOrLoadWeights(path)</c> forwarding to
/// <see cref="GetOrLoad" /> — so a pass names a project-relative checkpoint and gets weights loaded once
/// per checkpoint change rather than once per frame. Reloads happen only when the file stamp moves, so
/// retraining is picked up on the next iteration with no host restart.
///
/// The cache owns every instance it hands out. Callers must not dispose them.
///
/// Not thread-safe: it assumes the single-threaded render loop it is built for. A host driving passes
/// from several threads must serialize access.
/// </remarks>
public sealed class InferenceWeightsCache : IDisposable
{
    private readonly Dictionary<string, InferenceWeights> entries = new(StringComparer.Ordinal);
    private readonly string projectRoot;
    private bool disposed;

    /// <summary>
    /// Initializes a cache rooted at a project directory.
    /// </summary>
    /// <param name="projectRoot">
    /// The absolute project root. Relative paths resolve against it and may not escape it, which is what
    /// keeps a graph node's path parameter from reading arbitrary files.
    /// </param>
    public InferenceWeightsCache(string projectRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(projectRoot);
        this.projectRoot = Path.GetFullPath(projectRoot);
    }

    /// <summary>Gets the absolute project root relative paths resolve against.</summary>
    public string ProjectRoot => projectRoot;

    /// <summary>Gets the number of checkpoints currently held.</summary>
    public int Count => entries.Count;

    /// <summary>Gets the number of times a cached entry was reloaded because its file changed.</summary>
    /// <remarks>Exposed so a host can surface reload activity in pass diagnostics.</remarks>
    public int ReloadCount { get; private set; }

    /// <summary>
    /// Returns weights for a project-relative checkpoint, loaded once and reused across iterations and
    /// pass instances.
    /// </summary>
    /// <remarks>
    /// Reloads only when the file stamp changes. The cache owns the lifetime; the caller must not dispose
    /// the result, and must not hold it across a call that may reload it.
    /// </remarks>
    /// <param name="projectRelativePath">A checkpoint path relative to <see cref="ProjectRoot" />.</param>
    public InferenceWeights GetOrLoad(string projectRelativePath)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        var resolved = TrainingContext.ResolveWithinRoot(projectRoot, projectRelativePath);
        if (entries.TryGetValue(resolved, out var cached))
        {
            if (!cached.IsStale())
            {
                return cached;
            }

            entries.Remove(resolved);
            cached.Dispose();
            ReloadCount++;
        }

        var loaded = InferenceWeights.Load(resolved);
        entries.Add(resolved, loaded);
        return loaded;
    }

    /// <summary>Drops and disposes every cached checkpoint.</summary>
    public void Clear()
    {
        foreach (var weights in entries.Values)
        {
            weights.Dispose();
        }

        entries.Clear();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Clear();
        disposed = true;
    }
}
