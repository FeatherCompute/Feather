using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using Feather.NN;

namespace Feather.Blender.RenderHost;

internal static class TrainingRunner
{
    public static int Run(string requestPath, CancellationToken cancellationToken)
    {
        try
        {
            return RunCore(requestPath, cancellationToken);
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            WriteEvent("error", new { error = exception.GetType().Name, message = exception.Message });
            WriteEvent("finished", new
            {
                step = -1,
                finalLoss = (float?)null,
                reason = "failed",
                checkpointPath = (string?)null,
            });
            return 1;
        }
    }

    private static int RunCore(string requestPath, CancellationToken cancellationToken)
    {
        var request = TrainingRequest.Load(requestPath);
        var manifest = ProjectPassManifest.Load(request.ManifestPath);
        var definition = manifest.Trainers.SingleOrDefault(item =>
            string.Equals(item.TrainerGuid, request.TrainerGuid, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidDataException(
                $"Pass manifest does not define trainer GUID {request.TrainerGuid}.");
        if (!string.Equals(definition.TypeName, request.TypeName, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Training request type '{request.TypeName}' does not match manifest type '{definition.TypeName}'.");
        }

        var assemblyBytes = ReadFileShared(manifest.AssemblyPath);
        manifest.ValidateBuildId(assemblyBytes);
        var loadContext = new ProjectPassLoadContext(manifest.AssemblyPath);
        try
        {
            using var stream = new MemoryStream(assemblyBytes, writable: false);
            var assembly = loadContext.LoadFromStream(stream);
            var type = assembly.GetType(definition.TypeName, throwOnError: false, ignoreCase: false)
                ?? throw new InvalidDataException($"Trainer type '{definition.TypeName}' is not in the project assembly.");
            ValidateTrainerType(type, definition);
            using var job = (ITrainingJob)(Activator.CreateInstance(type)
                ?? throw new InvalidDataException($"Unable to create trainer '{definition.TypeName}'."));
            PassMemberBinder.BindParameters(job, request.Parameters);
            return RunLoop(requestPath, request, manifest.ProjectRoot, job, cancellationToken);
        }
        finally
        {
            loadContext.Unload();
        }
    }

    private static int RunLoop(
        string requestPath,
        ResolvedTrainingRequest request,
        string projectRoot,
        ITrainingJob job,
        CancellationToken cancellationToken)
    {
        var context = new TrainingContext(projectRoot, cancellationToken);
        var checkpointRelativePath = request.CheckpointDirectory.TrimEnd('/') + "/" +
            request.TrainerGuid + ".fthc";
        var checkpointPath = context.ResolveProjectPath(checkpointRelativePath);
        var clock = Stopwatch.StartNew();
        TrainingStepReport? latest = null;
        var lastStep = -1;
        var checkpointedStep = -1;
        var reason = "completed";

        WriteEvent("ready", new
        {
            requestPath = Path.GetFullPath(requestPath),
            trainerTypeName = request.TypeName,
            plannedSteps = request.PlannedSteps,
        });

        try
        {
            job.Initialize(context);
            for (var step = 0; step < request.PlannedSteps; step++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    reason = "cancelled";
                    break;
                }

                context.AdvanceTo(step);
                var report = job.Step(context);
                lastStep = step;
                if (report.IsReported)
                {
                    latest = report;
                }
                var reportNow = step == 0 || step + 1 == request.PlannedSteps ||
                    (step + 1) % request.ReportEverySteps == 0;
                if (reportNow && latest is { } measured)
                {
                    WriteEvent("loss", new
                    {
                        step,
                        plannedSteps = request.PlannedSteps,
                        loss = measured.Loss,
                        dispatchPath = measured.DispatchPath.ToString(),
                        stepsPerSecond = (step + 1) / System.Math.Max(clock.Elapsed.TotalSeconds, 0.000_001),
                        elapsedMilliseconds = clock.Elapsed.TotalMilliseconds,
                    });
                    if (measured.HasDiverged)
                    {
                        reason = "diverged";
                        break;
                    }
                }

                if ((step + 1) % request.CheckpointEverySteps == 0)
                {
                    SaveCheckpoint(job, checkpointPath, checkpointRelativePath, step, latest);
                    checkpointedStep = step;
                }
            }

            if (lastStep >= 0 && checkpointedStep != lastStep)
            {
                SaveCheckpoint(job, checkpointPath, checkpointRelativePath, lastStep, latest);
            }
            WriteEvent("finished", new
            {
                step = lastStep,
                finalLoss = latest?.Loss,
                reason,
                checkpointPath = checkpointRelativePath,
            });
            return reason == "diverged" ? 1 : 0;
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            WriteEvent("error", new { error = exception.GetType().Name, message = exception.Message });
            WriteEvent("finished", new
            {
                step = lastStep,
                finalLoss = latest?.Loss,
                reason = "failed",
                checkpointPath = checkpointRelativePath,
            });
            return 1;
        }
    }

    private static void SaveCheckpoint(
        ITrainingJob job,
        string absolutePath,
        string relativePath,
        int step,
        TrainingStepReport? latest)
    {
        var loss = latest?.Loss ?? float.NaN;
        Checkpoint.SaveAtomic(
            absolutePath,
            job.Parameters,
            new CheckpointMetadata(step, loss, ModelKind: job.GetType().FullName));
        WriteEvent("checkpoint", new
        {
            step,
            loss = latest?.Loss,
            path = relativePath,
            absolutePath,
            sizeInBytes = new FileInfo(absolutePath).Length,
        });
    }

    private static void ValidateTrainerType(Type type, ProjectTrainerDefinition definition)
    {
        if (!type.IsClass || type.IsAbstract || type.ContainsGenericParameters ||
            !typeof(ITrainingJob).IsAssignableFrom(type))
        {
            throw new InvalidDataException($"Trainer type '{type.FullName}' must be a concrete ITrainingJob class.");
        }
        if (type.GetConstructor(Type.EmptyTypes) is null)
        {
            throw new InvalidDataException($"Trainer type '{type.FullName}' must have a public parameterless constructor.");
        }
        var attribute = type.GetCustomAttribute<FeatherTrainerAttribute>(inherit: false)
            ?? throw new InvalidDataException($"Trainer type '{type.FullName}' has no FeatherTrainer attribute.");
        if (!Guid.TryParseExact(attribute.Guid, "D", out var guid) ||
            !string.Equals(guid.ToString("D"), definition.TrainerGuid, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Trainer type '{type.FullName}' does not declare manifest GUID {definition.TrainerGuid}.");
        }
    }

    private static byte[] ReadFileShared(string path)
    {
        using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        if (stream.Length > int.MaxValue)
        {
            throw new InvalidDataException($"Project artifact is too large to load: {path}");
        }
        var bytes = new byte[checked((int)stream.Length)];
        stream.ReadExactly(bytes);
        return bytes;
    }

    private static void WriteEvent(string name, object value)
        => Console.WriteLine(JsonSerializer.Serialize(new { @event = name, value }, ProtocolJson.Options));
}
