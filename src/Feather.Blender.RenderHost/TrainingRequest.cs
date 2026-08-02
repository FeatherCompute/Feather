using System.Text.Json;

namespace Feather.Blender.RenderHost;

internal sealed class TrainingRequest
{
    public const int CurrentSchemaVersion = 1;

    public int SchemaVersion { get; init; }
    public ulong RequestId { get; init; }
    public string ManifestPath { get; init; } = "";
    public string TrainerGuid { get; init; } = "";
    public string TypeName { get; init; } = "";
    public JsonElement Parameters { get; init; }
    public int PlannedSteps { get; init; }
    public int ReportEverySteps { get; init; } = 10;
    public int CheckpointEverySteps { get; init; } = 100;
    public string CheckpointDirectory { get; init; } = ".feather/training";

    public static ResolvedTrainingRequest Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var requestPath = Path.GetFullPath(path);
        TrainingRequest request;
        try
        {
            using var stream = new FileStream(
                requestPath, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
            request = JsonSerializer.Deserialize<TrainingRequest>(stream, ProtocolJson.Options)
                ?? throw new InvalidDataException("Training request JSON contains null.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"Training request JSON is invalid: {exception.Message}", exception);
        }

        request.Validate();
        var baseDirectory = Path.GetDirectoryName(requestPath)
            ?? throw new InvalidDataException("Training request has no parent directory.");
        return new ResolvedTrainingRequest(
            request.RequestId,
            Path.GetFullPath(Path.IsPathRooted(request.ManifestPath)
                ? request.ManifestPath
                : Path.Combine(baseDirectory, request.ManifestPath)),
            Guid.Parse(request.TrainerGuid).ToString("D"),
            request.TypeName,
            request.Parameters,
            request.PlannedSteps,
            request.ReportEverySteps,
            request.CheckpointEverySteps,
            request.CheckpointDirectory.Replace('\\', '/'));
    }

    private void Validate()
    {
        if (SchemaVersion != CurrentSchemaVersion)
        {
            throw new InvalidDataException($"Unsupported training request schema version: {SchemaVersion}.");
        }
        if (string.IsNullOrWhiteSpace(ManifestPath))
        {
            throw new InvalidDataException("Training request manifestPath is required.");
        }
        if (!Guid.TryParseExact(TrainerGuid, "D", out _))
        {
            throw new InvalidDataException("Training request trainerGuid must be a canonical GUID.");
        }
        if (string.IsNullOrWhiteSpace(TypeName))
        {
            throw new InvalidDataException("Training request typeName is required.");
        }
        if (PlannedSteps is < 1 or > 100_000_000)
        {
            throw new InvalidDataException("Training request plannedSteps must be between 1 and 100000000.");
        }
        if (ReportEverySteps is < 1 or > 1_000_000)
        {
            throw new InvalidDataException("Training request reportEverySteps must be between 1 and 1000000.");
        }
        if (CheckpointEverySteps is < 1 or > 100_000_000)
        {
            throw new InvalidDataException("Training request checkpointEverySteps must be between 1 and 100000000.");
        }
        if (string.IsNullOrWhiteSpace(CheckpointDirectory) || Path.IsPathRooted(CheckpointDirectory))
        {
            throw new InvalidDataException("Training request checkpointDirectory must be project-relative.");
        }
    }
}

internal sealed record ResolvedTrainingRequest(
    ulong RequestId,
    string ManifestPath,
    string TrainerGuid,
    string TypeName,
    JsonElement Parameters,
    int PlannedSteps,
    int ReportEverySteps,
    int CheckpointEverySteps,
    string CheckpointDirectory);
