using System.Text.Json.Nodes;

namespace Feather.Blender.RenderHost.Tests;

public sealed class ProjectPassManifestCacheTests
{
    [Fact]
    public void UnchangedManifestIsParsedOnceAndAtomicReplacementLoadsNewBuild()
    {
        var directory = Path.Combine(Path.GetTempPath(), $"feather-manifest-cache-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var manifestPath = Path.Combine(directory, "pass-manifest.json");
            WriteManifest(manifestPath, '0');
            var loads = 0;
            var cache = new ProjectPassManifestCache(path =>
            {
                loads++;
                return ProjectPassManifest.Load(path);
            });

            var first = cache.Load(manifestPath);
            var unchanged = cache.Load(manifestPath);

            Assert.Equal(1, loads);
            Assert.Same(first, unchanged);

            var replacementPath = Path.Combine(directory, "pass-manifest.replacement.json");
            WriteManifest(replacementPath, '1');
            File.SetLastWriteTimeUtc(replacementPath, DateTime.UtcNow.AddSeconds(1));
            File.Move(replacementPath, manifestPath, overwrite: true);

            var replacement = cache.Load(manifestPath);

            Assert.Equal(2, loads);
            Assert.NotSame(first, replacement);
            Assert.NotEqual(first.BuildId, replacement.BuildId);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void LoadsSupportedPassManifestVersions(int schemaVersion)
    {
        var directory = Path.Combine(Path.GetTempPath(), $"feather-manifest-version-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var manifestPath = Path.Combine(directory, "pass-manifest.json");
            WriteManifest(manifestPath, '0', schemaVersion);
            Assert.Single(ProjectPassManifest.Load(manifestPath).Passes);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void WriteManifest(string path, char buildIdCharacter, int schemaVersion = 1)
    {
        var document = new JsonObject
        {
            ["schemaVersion"] = schemaVersion,
            ["buildId"] = "sha256:" + new string(buildIdCharacter, 64),
            ["assemblyPath"] = "passes.dll",
            ["projectRoot"] = ".",
            ["passes"] = new JsonArray(
                new JsonObject
                {
                    ["passGuid"] = "20fc3bf1-eecb-41b1-bd03-d5c8a344ce6d",
                    ["typeName"] = "Test.Passes.MainPass",
                    ["inputs"] = new JsonArray(),
                    ["outputs"] = new JsonArray()
                })
        };
        File.WriteAllText(path, document.ToJsonString());
    }
}
