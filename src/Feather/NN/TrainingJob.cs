namespace Feather.NN;

/// <summary>
/// Marks a project type the platform can drive as a training job.
/// </summary>
/// <remarks>
/// The counterpart to <c>[FeatherPass]</c>. A host discovers the type by GUID and drives it one step
/// at a time; the attribute carries only identity and display metadata, exactly as the pass attribute
/// does, so a manifest writer can emit trainers alongside passes with the same shape.
/// </remarks>
/// <param name="guid">The persistent project identity of the trainer.</param>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class FeatherTrainerAttribute(string guid) : Attribute
{
    /// <summary>Gets the persistent project identity of the trainer.</summary>
    public string Guid { get; } = guid;

    /// <summary>Gets or sets the display name shown by a host's training panel.</summary>
    public string? Name { get; set; }

    /// <summary>Gets or sets the grouping a host's training panel files this trainer under.</summary>
    public string Category { get; set; } = "Training";

    /// <summary>Gets or sets the trainer's schema version.</summary>
    public int Version { get; set; } = 1;
}

/// <summary>
/// A project-authored training job driven one step at a time by a host.
/// </summary>
/// <remarks>
/// The loop is deliberately not in the SDK. A host owns cadence, reporting, checkpoint timing, and
/// cancellation, which is what lets a long-lived process interrupt training mid-run — a blocking
/// <c>Train(epochs)</c> could not be cancelled at all. Steps rather than epochs are the unit because
/// the AD path has no dataset concept yet, so "epoch" would have no meaning to enforce.
/// </remarks>
public interface ITrainingJob : IDisposable
{
    /// <summary>Gets the total planned steps, or 0 for open-ended training.</summary>
    int PlannedSteps { get; }

    /// <summary>
    /// Gets the parameters a host may checkpoint. Must be stable after <see cref="Initialize" />.
    /// </summary>
    /// <remarks>
    /// Stability matters because a host captures this list once and writes it repeatedly; a job that
    /// reallocated parameters mid-run would silently checkpoint dead tensors.
    /// </remarks>
    IReadOnlyList<IParameter> Parameters { get; }

    /// <summary>
    /// Allocates device resources, builds the AD kernel, and prepares the optimizer.
    /// </summary>
    /// <param name="context">The host-supplied context for this run.</param>
    void Initialize(TrainingContext context);

    /// <summary>
    /// Runs exactly one optimizer step and returns its report.
    /// </summary>
    /// <param name="context">The host-supplied context, carrying the step index about to run.</param>
    TrainingStepReport Step(TrainingContext context);
}

/// <summary>
/// One step's outcome.
/// </summary>
/// <remarks>
/// <paramref name="Loss" /> is <see cref="float.NaN" /> when the job chose not to read it back, which
/// is the expected case on the steps between a host's reporting cadence. A host distinguishes "not
/// measured" from "diverged" by checking <see cref="IsReported" /> before treating a non-finite loss
/// as divergence.
/// </remarks>
/// <param name="Step">The zero-based index of the step that ran.</param>
/// <param name="Loss">The step's loss, or NaN when the job did not read it back.</param>
/// <param name="DispatchPath">The native route the training dispatch took.</param>
public readonly record struct TrainingStepReport(
    int Step,
    float Loss,
    DispatchPath DispatchPath)
{
    /// <summary>Creates a report for a step whose loss was not read back.</summary>
    /// <param name="step">The zero-based index of the step that ran.</param>
    /// <param name="dispatchPath">The native route the training dispatch took.</param>
    public static TrainingStepReport Unreported(int step, DispatchPath dispatchPath)
        => new(step, float.NaN, dispatchPath);

    /// <summary>Creates a report for a step whose loss left the finite range.</summary>
    /// <param name="step">The zero-based index of the step that ran.</param>
    /// <param name="loss">The non-finite loss observed.</param>
    public static TrainingStepReport Diverged(int step, float loss)
        => new(step, loss, DispatchPath.None);

    /// <summary>Gets a value indicating whether this report carries a loss the host can display.</summary>
    public bool IsReported => !float.IsNaN(Loss);

    /// <summary>
    /// Gets a value indicating whether a reported loss left the finite range and training should stop.
    /// </summary>
    /// <remarks>
    /// False for an unreported step: a NaN placeholder means "nobody looked", not "the model blew up".
    /// </remarks>
    public bool HasDiverged => IsReported && !float.IsFinite(Loss);
}

/// <summary>
/// The host-supplied context a training job runs against, and the counterpart to a render pass's
/// render context.
/// </summary>
/// <remarks>
/// This is what keeps a job host-agnostic: the project root, cancellation, the step index, and
/// hyperparameters all arrive from the host rather than being read from the environment. A console
/// sample constructs one directly; a platform host constructs one per run from a request file.
///
/// The type is mutable in exactly one respect — a host advances <see cref="Step" /> through
/// <see cref="AdvanceTo" /> rather than reallocating the context per step, because the loss stream
/// and settings are run-scoped rather than step-scoped.
/// </remarks>
public sealed class TrainingContext
{
    private readonly Dictionary<string, object> settings;
    private readonly Action<TrainingStepReport>? lossStream;

    /// <summary>
    /// Initializes a training context.
    /// </summary>
    /// <param name="projectRoot">The absolute project root a host resolves relative paths against.</param>
    /// <param name="cancellationToken">Cooperative cancellation owned by the host.</param>
    /// <param name="settings">Host-supplied hyperparameters, typically from graph node parameters.</param>
    /// <param name="lossStream">
    /// An optional sink a job may push reports into. A host normally leaves this null and reports from
    /// its own loop, but a job that runs several inner iterations per step can surface them here.
    /// </param>
    public TrainingContext(
        string projectRoot,
        CancellationToken cancellationToken = default,
        IReadOnlyDictionary<string, object>? settings = null,
        Action<TrainingStepReport>? lossStream = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(projectRoot);
        ProjectRoot = Path.GetFullPath(projectRoot);
        CancellationToken = cancellationToken;
        this.settings = settings is null
            ? new Dictionary<string, object>(StringComparer.Ordinal)
            : new Dictionary<string, object>(settings, StringComparer.Ordinal);
        this.lossStream = lossStream;
    }

    /// <summary>Gets the absolute project root, from the pass manifest's project root.</summary>
    public string ProjectRoot { get; }

    /// <summary>Gets cooperative cancellation owned by the host.</summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>Gets the zero-based index of the step about to run.</summary>
    public int Step { get; private set; }

    /// <summary>Sets the index of the step about to run.</summary>
    /// <remarks>
    /// Called by the host's loop before <see cref="ITrainingJob.Step" />. Monotonic: a host cannot
    /// rewind the counter, because a job may have already checkpointed under the higher number.
    /// </remarks>
    /// <param name="step">The zero-based step index about to run.</param>
    public void AdvanceTo(int step)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(step);
        if (step < Step)
        {
            throw new ArgumentOutOfRangeException(nameof(step), $"Training step cannot move backwards from {Step} to {step}.");
        }

        Step = step;
    }

    /// <summary>Pushes a step report into the host's loss stream, if one was supplied.</summary>
    /// <param name="report">The report to publish.</param>
    public void ReportLoss(TrainingStepReport report)
        => lossStream?.Invoke(report);

    /// <summary>Reads a host-supplied hyperparameter.</summary>
    /// <typeparam name="T">The unmanaged setting type.</typeparam>
    /// <param name="name">The setting name, matching a graph node parameter name.</param>
    /// <param name="value">The setting value when present.</param>
    /// <returns>True when a setting of the requested name and type exists.</returns>
    public bool TryGetSetting<T>(string name, out T value)
        where T : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (settings.TryGetValue(name, out var stored) && stored is T typed)
        {
            value = typed;
            return true;
        }

        value = default;
        return false;
    }

    /// <summary>Reads a host-supplied hyperparameter, falling back to a default.</summary>
    /// <typeparam name="T">The unmanaged setting type.</typeparam>
    /// <param name="name">The setting name, matching a graph node parameter name.</param>
    /// <param name="fallback">The value to use when the host supplied none.</param>
    public T GetSetting<T>(string name, T fallback)
        where T : unmanaged
        => TryGetSetting<T>(name, out var value) ? value : fallback;

    /// <summary>
    /// Resolves a project-relative path and creates its parent directory.
    /// </summary>
    /// <remarks>
    /// Deliberately on the context rather than a static helper, so the host owns the sandbox root.
    /// Rooted paths and paths that escape <see cref="ProjectRoot" /> after normalization are rejected,
    /// which is what stops a graph node's path parameter from writing anywhere on the machine.
    /// </remarks>
    /// <param name="relativePath">A path relative to <see cref="ProjectRoot" />.</param>
    /// <returns>The absolute path, with its parent directory created.</returns>
    public string ResolveProjectPath(string relativePath)
    {
        var resolved = ResolveProjectPathWithoutCreating(relativePath);
        var directory = Path.GetDirectoryName(resolved);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        return resolved;
    }

    /// <summary>
    /// Resolves a project-relative path without creating any directory.
    /// </summary>
    /// <param name="relativePath">A path relative to <see cref="ProjectRoot" />.</param>
    public string ResolveProjectPathWithoutCreating(string relativePath)
        => ResolveWithinRoot(ProjectRoot, relativePath);

    /// <summary>
    /// Resolves <paramref name="relativePath" /> under <paramref name="root" />, rejecting escapes.
    /// </summary>
    /// <remarks>
    /// Shared with the inference-weights cache so both halves of the platform apply one sandbox rule.
    /// </remarks>
    /// <param name="root">The absolute directory the path must stay inside.</param>
    /// <param name="relativePath">The relative path to resolve.</param>
    internal static string ResolveWithinRoot(string root, string relativePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(relativePath);
        if (Path.IsPathRooted(relativePath))
        {
            throw new ArgumentException($"'{relativePath}' must be relative to the project root.", nameof(relativePath));
        }

        var resolved = Path.GetFullPath(Path.Combine(root, relativePath));
        var rootWithSeparator = root.EndsWith(Path.DirectorySeparatorChar)
            ? root
            : root + Path.DirectorySeparatorChar;
        if (!resolved.StartsWith(rootWithSeparator, StringComparison.Ordinal))
        {
            throw new ArgumentException($"'{relativePath}' resolves outside the project root '{root}'.", nameof(relativePath));
        }

        return resolved;
    }
}
