using System.Diagnostics;
using System.Text.Json;

namespace Feather.Generator.Tests;

public class RenderGraphBuildTests
{
    /// <summary>C-class build assertion deadline; defaults to 120 seconds.</summary>
    private static TimeSpan TestTimeout
    {
        get
        {
            const string name = "FEATHER_GENERATOR_TEST_TIMEOUT_SECONDS";
            var value = Environment.GetEnvironmentVariable(name);
            if (value is null)
            {
                return TimeSpan.FromSeconds(120);
            }
            if (!double.TryParse(
                    value,
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out var seconds) ||
                !double.IsFinite(seconds) ||
                seconds <= 0)
            {
                throw new InvalidOperationException($"{name} must be a positive number of seconds.");
            }
            return TimeSpan.FromSeconds(seconds);
        }
    }

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
            Assert.Equal(2, root.GetProperty("schemaVersion").GetInt32());
            var firstBuildId = root.GetProperty("buildId").GetString();
            Assert.Matches("^sha256:[0-9a-f]{64}$", firstBuildId);
            Assert.Equal(expectedAssemblyPath, root.GetProperty("assemblyPath").GetString());
            Assert.Equal("Generated/feather-ir", root.GetProperty("feirDirectory").GetString());
            Assert.Equal(
                Path.GetRelativePath(outputDirectory, sampleDirectory).Replace('\\', '/'),
                root.GetProperty("projectRoot").GetString());
            Assert.True(File.Exists(Path.Combine(sampleDirectory, expectedAssemblyPath.Replace('/', Path.DirectorySeparatorChar))));

            var pass = Assert.Single(root.GetProperty("passes").EnumerateArray());
            var passGuid = pass.GetProperty("passGuid").GetString();
            Assert.Equal("01c671a1-9b4e-5cab-b7e1-c101348af596", passGuid);
            Assert.Equal("BlenderRenderGraph.Passes.MinimalRasterPass", pass.GetProperty("typeName").GetString());
            Assert.Equal(expectedAssemblyPath, pass.GetProperty("assemblyPath").GetString());
            Assert.Equal(string.Empty, pass.GetProperty("feirPath").GetString());
            Assert.Equal("Passes/MinimalRasterPass.cs", pass.GetProperty("source").GetProperty("path").GetString());
            Assert.Equal("RGBA8", Assert.Single(pass.GetProperty("outputs").EnumerateArray()).GetProperty("format").GetString());

            var parameters = pass.GetProperty("parameters")
                .EnumerateArray()
                .ToDictionary(item => item.GetProperty("name").GetString()!);
            Assert.Equal(2, parameters.Count);
            var exposure = parameters["Exposure"];
            Assert.Equal(1.0, exposure.GetProperty("defaultValue").GetDouble());
            Assert.Equal(0.0, exposure.GetProperty("min").GetDouble());
            Assert.Equal(8.0, exposure.GetProperty("max").GetDouble());
            var viewMode = parameters["ViewMode"];
            Assert.Equal(0, viewMode.GetProperty("defaultValue").GetInt32());
            Assert.Equal(0.0, viewMode.GetProperty("min").GetDouble());
            Assert.Equal(2.0, viewMode.GetProperty("max").GetDouble());

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

    [Fact]
    public async Task AssetContractSampleBuildsAndExportsBoundedRelativeManifest()
    {
        var repositoryRoot = FindRepositoryRoot();
        var sampleDirectory = Path.Combine(repositoryRoot, "samples", "AssetContracts");
        var projectPath = Path.Combine(sampleDirectory, "AssetContracts.csproj");
        var outputDirectory = Path.Combine(
            Path.GetTempPath(),
            "FeatherAssetBuildTests",
            Guid.NewGuid().ToString("N"));
        var passManifestPath = Path.Combine(outputDirectory, "pass-manifest.json");
        var assetManifestPath = Path.Combine(outputDirectory, "asset-manifest.json");
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
                passManifestPath,
                version: "1.0.0",
                assetManifestPath);
            Assert.False(File.Exists(passManifestPath));
            Assert.True(File.Exists(assetManifestPath), $"Asset manifest was not written:{Environment.NewLine}{output}");

            var manifestText = await File.ReadAllTextAsync(assetManifestPath);
            Assert.DoesNotContain(repositoryRoot, manifestText, StringComparison.Ordinal);
            using var manifest = JsonDocument.Parse(manifestText);
            var root = manifest.RootElement;
            Assert.Equal(1, root.GetProperty("schemaVersion").GetInt32());
            Assert.Equal("Feather.AssetAssemblyManifest", root.GetProperty("kind").GetString());
            Assert.Matches("^sha256:[0-9a-f]{64}$", root.GetProperty("generatorManifestHash").GetString());
            Assert.Matches("^sha256:[0-9a-f]{64}$", root.GetProperty("assemblyHash").GetString());
            Assert.Matches("^sha256:[0-9a-f]{64}$", root.GetProperty("toolchainHash").GetString());
            Assert.Matches("^sha256:[0-9a-f]{64}$", root.GetProperty("buildId").GetString());
            Assert.Matches("^sha256:[0-9a-f]{64}$", root.GetProperty("manifestHash").GetString());
            Assert.Equal(
                $"bin/{configuration}/net10.0/AssetContracts.dll",
                root.GetProperty("assemblyPath").GetString());

            var types = root.GetProperty("assetTypes").EnumerateArray().ToArray();
            Assert.Equal(3, types.Length);
            var gradient = types.Single(type =>
                type.GetProperty("typeId").GetString() == "878827ac-7fe1-4990-acad-554923b696c8");
            var source = gradient.GetProperty("source");
            Assert.Equal("GradientFieldAssets.cs", source.GetProperty("path").GetString());
            Assert.Matches("^sha256:[0-9a-f]{64}$", source.GetProperty("documentHash").GetString());
            var noPreview = types.Single(type =>
                type.GetProperty("typeId").GetString() == "97c5237c-2ee8-4858-8881-f5e4726116da");
            Assert.Empty(noPreview.GetProperty("capabilities").EnumerateArray());
            Assert.Empty(noPreview.GetProperty("productSlots").EnumerateArray());
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
        string version,
        string? assetManifestPath = null)
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
        if (assetManifestPath is not null)
        {
            startInfo.ArgumentList.Add($"--property:FeatherAssetManifestPath={assetManifestPath}");
        }
        startInfo.ArgumentList.Add($"--property:Version={version}");

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Unable to start dotnet build.");
        var standardOutput = process.StandardOutput.ReadToEndAsync();
        var standardError = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TestTimeout);
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
