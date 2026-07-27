using System.Diagnostics;
using System.Text.Json;

namespace Feather.Generator.Tests;

public class RenderGraphBuildTests
{
    [Fact]
    public async Task LocalProjectReferenceSampleBuildsAndExportsManifest()
    {
        var repositoryRoot = FindRepositoryRoot();
        var sampleDirectory = Path.Combine(repositoryRoot, "samples", "BlenderRenderGraph");
        var projectPath = Path.Combine(sampleDirectory, "BlenderRenderGraph.csproj");
        var outputDirectory = Path.Combine(
            Path.GetTempPath(),
            "FeatherRenderGraphBuildTests",
            Guid.NewGuid().ToString("N"));
        var manifestPath = Path.Combine(outputDirectory, "pass-manifest.json");
        var configuration = Directory.GetParent(typeof(RenderGraphBuildTests).Assembly.Location)!
            .Parent!
            .Name;

        Directory.CreateDirectory(outputDirectory);
        try
        {
            var output = await BuildSampleAsync(
                repositoryRoot,
                projectPath,
                configuration,
                manifestPath,
                version: "1.0.0");
            Assert.True(File.Exists(manifestPath), $"Manifest was not written:{Environment.NewLine}{output}");

            using var manifest = JsonDocument.Parse(await File.ReadAllTextAsync(manifestPath));
            var root = manifest.RootElement;
            var expectedAssemblyPath = $"bin/{configuration}/net10.0/BlenderRenderGraph.dll";
            Assert.Equal(1, root.GetProperty("schemaVersion").GetInt32());
            var firstBuildId = root.GetProperty("buildId").GetString();
            Assert.Matches("^sha256:[0-9a-f]{64}$", firstBuildId);
            Assert.Equal(expectedAssemblyPath, root.GetProperty("assemblyPath").GetString());
            Assert.Equal("Generated/feather-ir", root.GetProperty("feirDirectory").GetString());
            Assert.True(File.Exists(Path.Combine(sampleDirectory, expectedAssemblyPath.Replace('/', Path.DirectorySeparatorChar))));

            var pass = Assert.Single(root.GetProperty("passes").EnumerateArray());
            var passGuid = pass.GetProperty("passGuid").GetString();
            Assert.Equal("01c671a1-9b4e-5cab-b7e1-c101348af596", passGuid);
            Assert.Equal("BlenderRenderGraph.Passes.MinimalRasterPass", pass.GetProperty("typeName").GetString());
            Assert.Equal(expectedAssemblyPath, pass.GetProperty("assemblyPath").GetString());
            Assert.Equal($"Generated/feather-ir/{passGuid}.feir", pass.GetProperty("feirPath").GetString());
            Assert.Equal("Passes/MinimalRasterPass.cs", pass.GetProperty("source").GetProperty("path").GetString());
            Assert.Equal("RGBA16Float", Assert.Single(pass.GetProperty("outputs").EnumerateArray()).GetProperty("format").GetString());

            var parameter = Assert.Single(pass.GetProperty("parameters").EnumerateArray());
            Assert.Equal("Exposure", parameter.GetProperty("name").GetString());
            Assert.Equal(1.0, parameter.GetProperty("defaultValue").GetDouble());
            Assert.Equal(0.0, parameter.GetProperty("min").GetDouble());
            Assert.Equal(8.0, parameter.GetProperty("max").GetDouble());

            await BuildSampleAsync(
                repositoryRoot,
                projectPath,
                configuration,
                manifestPath,
                version: "1.0.1");
            using var rebuiltManifest = JsonDocument.Parse(await File.ReadAllTextAsync(manifestPath));
            var secondBuildId = rebuiltManifest.RootElement.GetProperty("buildId").GetString();
            Assert.Matches("^sha256:[0-9a-f]{64}$", secondBuildId);
            Assert.NotEqual(firstBuildId, secondBuildId);
        }
        finally
        {
            Directory.Delete(outputDirectory, recursive: true);
        }
    }

    private static async Task<string> BuildSampleAsync(
        string repositoryRoot,
        string projectPath,
        string configuration,
        string manifestPath,
        string version)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ?? "dotnet",
            WorkingDirectory = repositoryRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        startInfo.Environment["MSBUILDDISABLENODEREUSE"] = "1";
        startInfo.ArgumentList.Add("build");
        startInfo.ArgumentList.Add(projectPath);
        startInfo.ArgumentList.Add("--nologo");
        startInfo.ArgumentList.Add("--disable-build-servers");
        startInfo.ArgumentList.Add("--no-incremental");
        startInfo.ArgumentList.Add("--configuration");
        startInfo.ArgumentList.Add(configuration);
        startInfo.ArgumentList.Add("--property:BuildProjectReferences=false");
        startInfo.ArgumentList.Add($"--property:FeatherPassManifestPath={manifestPath}");
        startInfo.ArgumentList.Add($"--property:Version={version}");

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Unable to start dotnet build.");
        var standardOutput = process.StandardOutput.ReadToEndAsync();
        var standardError = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(2));
        try
        {
            await process.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException)
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException("Timed out building the BlenderRenderGraph sample.");
        }

        var output = await standardOutput.WaitAsync(timeout.Token);
        var error = await standardError.WaitAsync(timeout.Token);
        Assert.True(
            process.ExitCode == 0,
            $"dotnet build failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{Environment.NewLine}{error}");
        return output;
    }

    private static string FindRepositoryRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory);
             directory is not null;
             directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Feather.slnx")))
            {
                return directory.FullName;
            }
        }

        throw new DirectoryNotFoundException("Unable to find the Feather repository root.");
    }
}
