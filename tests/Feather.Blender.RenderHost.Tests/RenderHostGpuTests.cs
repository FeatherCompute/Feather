using System.Buffers.Binary;
using System.Diagnostics;
using System.Security;

namespace Feather.Blender.RenderHost.Tests;

public sealed class RenderHostGpuTests
{
    private const string HotReloadPassType = "HotReloadProject.Passes.MinimalRasterPass";

    [Theory]
    [Trait("Category", "Gpu")]
    [InlineData(1)]
    [InlineData(4)]
    public void LegacyRequestWithoutManifestUsesBuiltInRasterCompatibilityPath(int sampleCount)
    {
        using var fixture = new ProtocolFixture();
        fixture.WriteScene();
        fixture.WriteGraph(sampleCount: sampleCount);
        fixture.WriteRequest();
        using var host = new RenderHostRunner();

        var result = host.RenderOnce(fixture.RequestPath);

        Assert.Equal("TypedEasyGpu", result.DispatchPath);
        Assert.Equal(1, result.TriangleCount);
        Assert.Equal(3, result.VertexCount);
        var frame = File.ReadAllBytes(fixture.OutputPath);
        Assert.Equal(42ul, BinaryPrimitives.ReadUInt64LittleEndian(frame.AsSpan(32, 8)));
        var pixels = frame.AsSpan(40);
        var brightPixelCount = 0;
        var backgroundPixelCount = 0;
        for (var offset = 0; offset < pixels.Length; offset += 4)
        {
            if (pixels[offset] > 200 && pixels[offset + 1] > 200 && pixels[offset + 2] > 200)
            {
                brightPixelCount++;
            }
            if (pixels[offset] < 10 && pixels[offset + 1] < 10 && pixels[offset + 2] < 20)
            {
                backgroundPixelCount++;
            }
        }

        Assert.True(brightPixelCount > 300, $"Expected rendered triangle pixels, found {brightPixelCount}.");
        Assert.True(backgroundPixelCount > 300, $"Expected background pixels, found {backgroundPixelCount}.");
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public async Task RebuiltProjectAssemblyReloadsSamePassTypeAndChangesShaderOutput()
    {
        using var fixture = new ProtocolFixture();
        var project = new HotReloadProject(fixture.Root, FindRepositoryRoot());
        await project.BuildAsync("1.0f, 0.05f, 0.05f", buildNumber: 1);
        fixture.WriteScene();
        fixture.WriteGraph(typeName: HotReloadPassType);
        fixture.WriteRequest(manifestPath: project.ManifestPath);
        using var host = new RenderHostRunner();

        var first = host.RenderOnce(fixture.RequestPath);
        var firstPixels = File.ReadAllBytes(fixture.OutputPath)[40..];
        var firstChannels = SumRedAndBlue(firstPixels);

        Assert.True(first.PassReloaded);
        Assert.Equal(HotReloadPassType, first.PassType);
        Assert.Equal("TypedEasyGpu", first.DispatchPath);
        Assert.True(
            firstChannels.Red > firstChannels.Blue * 3,
            $"Expected a red shader result, got R={firstChannels.Red}, B={firstChannels.Blue}.");

        await project.BuildAsync("0.05f, 0.05f, 1.0f", buildNumber: 2);
        var second = host.RenderOnce(fixture.RequestPath);
        var secondPixels = File.ReadAllBytes(fixture.OutputPath)[40..];
        var secondChannels = SumRedAndBlue(secondPixels);

        Assert.True(second.PassReloaded);
        Assert.NotEqual(first.BuildId, second.BuildId);
        Assert.Equal(HotReloadPassType, second.PassType);
        Assert.True(
            secondChannels.Blue > secondChannels.Red * 3,
            $"Expected a blue shader result, got R={secondChannels.Red}, B={secondChannels.Blue}.");
        Assert.NotNull(host.LastUnloadedPassContextForTesting);
        AssertEventuallyUnloaded(host.LastUnloadedPassContextForTesting!);

        var third = host.RenderOnce(fixture.RequestPath);
        Assert.False(third.PassReloaded);
        Assert.Equal(second.BuildId, third.BuildId);
    }

    private static (long Red, long Blue) SumRedAndBlue(ReadOnlySpan<byte> pixels)
    {
        long red = 0;
        long blue = 0;
        for (var offset = 0; offset < pixels.Length; offset += 4)
        {
            red += pixels[offset];
            blue += pixels[offset + 2];
        }
        return (red, blue);
    }

    private static void AssertEventuallyUnloaded(WeakReference reference)
    {
        for (var attempt = 0; attempt < 10 && reference.IsAlive; attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
        Assert.False(reference.IsAlive, "The previous shader AssemblyLoadContext remained alive.");
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

    private sealed class HotReloadProject
    {
        private readonly string projectDirectory;
        private readonly string sourcePath;
        private readonly string template;
        private readonly string configuration;

        public HotReloadProject(string parentDirectory, string repositoryRoot)
        {
            projectDirectory = Path.Combine(parentDirectory, "HotReloadProject");
            sourcePath = Path.Combine(projectDirectory, "MinimalRasterPass.cs");
            ManifestPath = Path.Combine(projectDirectory, "Generated", "pass-manifest.json");
            Directory.CreateDirectory(projectDirectory);

            var featherProject = SecurityElement.Escape(
                Path.Combine(repositoryRoot, "src", "Feather", "Feather.csproj"))!;
            var generatorProject = SecurityElement.Escape(
                Path.Combine(repositoryRoot, "src", "Feather.Generators", "Feather.Generators.csproj"))!;
            var targets = SecurityElement.Escape(
                Path.Combine(repositoryRoot, "src", "Feather", "build", "FeatherCompute.targets"))!;
            File.WriteAllText(
                Path.Combine(projectDirectory, "HotReloadProject.csproj"),
                $$"""
                <Project Sdk="Microsoft.NET.Sdk">
                  <PropertyGroup>
                    <TargetFramework>net10.0</TargetFramework>
                    <ImplicitUsings>enable</ImplicitUsings>
                    <Nullable>enable</Nullable>
                  </PropertyGroup>
                  <ItemGroup>
                    <ProjectReference Include="{{featherProject}}" />
                    <ProjectReference Include="{{generatorProject}}"
                                      OutputItemType="Analyzer"
                                      ReferenceOutputAssembly="false" />
                  </ItemGroup>
                  <Import Project="{{targets}}" />
                </Project>
                """);
            template = File.ReadAllText(Path.Combine(
                repositoryRoot,
                "tests",
                "Feather.Blender.RenderHost.Tests",
                "HotReloadPassTemplate.txt"));
            configuration = Directory.GetParent(typeof(RenderHostGpuTests).Assembly.Location)!
                .Parent!
                .Name;
        }

        public string ManifestPath { get; }

        public async Task BuildAsync(string tint, int buildNumber)
        {
            File.WriteAllText(
                sourcePath,
                template.Replace("__TINT__", tint, StringComparison.Ordinal));
            var startInfo = new ProcessStartInfo
            {
                FileName = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ?? "dotnet",
                WorkingDirectory = projectDirectory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            startInfo.Environment["MSBUILDDISABLENODEREUSE"] = "1";
            startInfo.ArgumentList.Add("build");
            startInfo.ArgumentList.Add("HotReloadProject.csproj");
            startInfo.ArgumentList.Add("--nologo");
            startInfo.ArgumentList.Add("--disable-build-servers");
            startInfo.ArgumentList.Add("--no-incremental");
            startInfo.ArgumentList.Add("--configuration");
            startInfo.ArgumentList.Add(configuration);
            startInfo.ArgumentList.Add("--property:BuildProjectReferences=false");
            startInfo.ArgumentList.Add($"--property:Version=1.0.{buildNumber}");

            using var process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Unable to start the hot-reload project build.");
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
                throw new TimeoutException("Timed out building the hot-reload pass project.");
            }

            var output = await standardOutput.WaitAsync(timeout.Token);
            var error = await standardError.WaitAsync(timeout.Token);
            Assert.True(
                process.ExitCode == 0,
                $"Hot-reload project build failed:{Environment.NewLine}{output}{Environment.NewLine}{error}");
            Assert.True(File.Exists(ManifestPath), "The hot-reload build did not publish its pass manifest.");
        }
    }
}
