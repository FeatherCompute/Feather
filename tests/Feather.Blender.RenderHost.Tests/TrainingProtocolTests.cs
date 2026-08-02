using System.Text.Json;

namespace Feather.Blender.RenderHost.Tests;

public sealed class TrainingProtocolTests
{
    [Fact]
    public void TrainingRequestResolvesManifestAndPreservesProjectRelativeCheckpointDirectory()
    {
        var directory = Path.Combine(Path.GetTempPath(), $"feather-training-request-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var path = Path.Combine(directory, "train.request.json");
            File.WriteAllText(path, JsonSerializer.Serialize(new
            {
                schemaVersion = 1,
                requestId = 7,
                manifestPath = "../../Generated/pass-manifest.json",
                trainerGuid = "9d4f8a2e-6c31-4b75-a920-1e3d7f5c8b64",
                typeName = "Example.NeuralTrainer",
                parameters = Array.Empty<object>(),
                plannedSteps = 300,
                reportEverySteps = 10,
                checkpointEverySteps = 100,
                checkpointDirectory = ".feather/training/",
            }));

            var request = TrainingRequest.Load(path);

            Assert.Equal(300, request.PlannedSteps);
            Assert.Equal(10, request.ReportEverySteps);
            Assert.Equal(100, request.CheckpointEverySteps);
            Assert.Equal(".feather/training/", request.CheckpointDirectory);
            Assert.Equal(
                Path.GetFullPath(Path.Combine(directory, "../../Generated/pass-manifest.json")),
                request.ManifestPath);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Theory]
    [InlineData("/tmp/training")]
    [InlineData("")]
    public void TrainingRequestRejectsNonRelativeCheckpointDirectory(string checkpointDirectory)
    {
        var path = Path.Combine(Path.GetTempPath(), $"feather-training-request-{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(path, JsonSerializer.Serialize(new
            {
                schemaVersion = 1,
                manifestPath = "manifest.json",
                trainerGuid = "9d4f8a2e-6c31-4b75-a920-1e3d7f5c8b64",
                typeName = "Example.NeuralTrainer",
                plannedSteps = 1,
                reportEverySteps = 1,
                checkpointEverySteps = 1,
                checkpointDirectory,
            }));

            Assert.Throws<InvalidDataException>(() => TrainingRequest.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void RenderHostOptionsExposeMutuallyExclusiveTrainingMode()
    {
        var options = RenderHostOptions.Parse(["--request", "train.json", "--train"]);

        Assert.True(options.Train);
        Assert.False(options.Watch);
        Assert.Throws<ArgumentException>(() => RenderHostOptions.Parse(
            ["--request", "train.json", "--train", "--watch"]));
    }
}
