namespace Feather.Integration.Tests;

using System.Reflection;
using System.Text.RegularExpressions;
using System.Xml.Linq;

public class ProjectShapeTests
{
    [Fact]
    public void RepositoryContainsRequiredTopLevelWorkstreams()
    {
        var root = FindRepositoryRoot();

        Assert.True(Directory.Exists(Path.Combine(root, "src", "Feather")));
        Assert.True(Directory.Exists(Path.Combine(root, "src", "Feather.Native")));
        Assert.True(Directory.Exists(Path.Combine(root, "src", "Feather.Generators")));
        Assert.True(Directory.Exists(Path.Combine(root, "native")));
        Assert.True(Directory.Exists(Path.Combine(root, "docs")));
    }

    [Fact]
    public void NativeAssetsProjectPacksRidNativeFiles()
    {
        var root = FindRepositoryRoot();
        var projectPath = Path.Combine(root, "src", "Feather.NativeAssets", "Feather.NativeAssets.csproj");
        var project = XDocument.Load(projectPath);
        var text = File.ReadAllText(projectPath);
        var nativeAssetItem = project.Descendants("None")
            .SingleOrDefault(element => (string?)element.Attribute("Include") == "$(NativeAssetStagingRoot)runtimes/**/native/*");

        Assert.NotNull(nativeAssetItem);
        Assert.Equal("true", (string?)nativeAssetItem.Attribute("Pack"));
        Assert.Equal("runtimes", (string?)nativeAssetItem.Attribute("LinkBase"));
        Assert.Equal("%(LinkBase)/%(RecursiveDir)%(Filename)%(Extension)", (string?)nativeAssetItem.Attribute("PackagePath"));
        Assert.Contains("RuntimeIdentifiers", text);
        Assert.Contains("NativeAssetStagingRoot", text);
        Assert.Contains("../../artifacts/native-assets/", text);
        Assert.Contains("runtimes/**/native/*", text);
        Assert.DoesNotContain("CopyLocalNativeRuntimeAsset", text);
        Assert.DoesNotContain("src/Feather.NativeAssets/runtimes", text);
    }

    [Fact]
    public void ShaderLibraryPacksWhereTheTargetsFileLooksForIt()
    {
        // The two halves of shader injection live in different files and are only wrong together:
        // Feather.csproj decides where the sources land in the package, FeatherCompute.targets decides
        // where to glob for them, and the targets file resolves that path relative to its own
        // directory. NuGet imports it from buildTransitive, so the sources have to be there too.
        //
        // Source mode reads the sources straight out of the checkout, so it keeps working either way.
        // That asymmetry is what makes this worth a test: nothing in a local build or in the generator
        // suite notices, and the first symptom is a consumer whose project cannot resolve a shared
        // helper against a published package.
        var root = FindRepositoryRoot();
        var project = XDocument.Load(Path.Combine(root, "src", "Feather", "Feather.csproj"));
        var targets = File.ReadAllText(
            Path.Combine(root, "src", "Feather", "build", "FeatherCompute.targets"));

        var shaderItem = project.Descendants("None")
            .Single(element => (string?)element.Attribute("Include") == @"Shaders\**\*.cs");
        var targetsItem = project.Descendants("None")
            .Single(element => (string?)element.Attribute("Include") == @"build\FeatherCompute.targets");

        Assert.Equal("true", (string?)shaderItem.Attribute("Pack"));
        Assert.Equal(@"buildTransitive\shaders", (string?)shaderItem.Attribute("PackagePath"));

        // Read the directory the targets file falls back to out of the targets file itself, so the two
        // cannot drift apart without one of these assertions failing.
        var packagedTargetsDirectory = Path.GetDirectoryName(
            ((string?)targetsItem.Attribute("PackagePath"))!.Replace('\\', '/'));
        var packagedShaderDirectory = ((string?)shaderItem.Attribute("PackagePath"))!.Replace('\\', '/');

        Assert.Contains(
            "<FeatherShaderLibraryDirectory Condition=\"'$(FeatherShaderLibraryDirectory)' == ''\">$(MSBuildThisFileDirectory)shaders</FeatherShaderLibraryDirectory>",
            targets);
        Assert.Equal($"{packagedTargetsDirectory}/shaders", packagedShaderDirectory);

        // The helpers are only shareable as source; a compiled copy is rejected with FE0008.
        Assert.Contains(@"<Compile Remove=""Shaders\**\*.cs"" />", File.ReadAllText(
            Path.Combine(root, "src", "Feather", "Feather.csproj")), StringComparison.Ordinal);
    }

    [Fact]
    public void NativeBuildLinksLuisaRuntimeAndExcludesLegacyRuntime()
    {
        var root = FindRepositoryRoot();
        var cmakePath = Path.Combine(root, "native", "CMakeLists.txt");
        var text = File.ReadAllText(cmakePath);

        Assert.Contains("add_subdirectory(\"${CMAKE_CURRENT_SOURCE_DIR}/../LuisaCompute\"", text);
        Assert.Contains("target_link_libraries(feather PRIVATE luisa-compute-runtime luisa-compute-xir luisa-compute-dsl)", text);
        Assert.DoesNotContain("EasyGPU", text, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeReferenceCoverageTestsAreQuarantined()
    {
        var fallbackTests = typeof(ProjectShapeTests).Assembly
            .GetTypes()
            .SelectMany(type => type.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                .Where(method => method.GetCustomAttribute<FactAttribute>() is not null)
                .Where(method => type.Name.Contains("Fallback", StringComparison.Ordinal)
                    || method.Name.Contains("Fallback", StringComparison.Ordinal))
                .Select(method => new { Type = type, Method = method }))
            .ToArray();

        Assert.NotEmpty(fallbackTests);
        foreach (var test in fallbackTests)
        {
            var hasNativeReferenceTrait = test.Method.GetCustomAttributesData()
                .Concat(test.Type.GetCustomAttributesData())
                .Any(IsNativeReferenceFallbackTrait);

            Assert.True(
                hasNativeReferenceTrait,
                $"{test.Type.FullName}.{test.Method.Name} must be marked as native reference fallback coverage, not DSL completion proof.");
        }
    }

    [Fact]
    public void FeatherNnIndustrializedSurfaceDoesNotExposeTrainerInternalsOrAmbiguousHostInference()
    {
        var root = FindRepositoryRoot();
        var sequenceModels = File.ReadAllText(Path.Combine(root, "src", "Feather", "NN", "SequenceModels.cs"));
        var nnSources = Directory.GetFiles(Path.Combine(root, "src", "Feather", "NN"), "*.cs", SearchOption.AllDirectories)
            .Select(File.ReadAllText)
            .ToArray();
        var sampleAndTestSources = Directory.GetFiles(Path.Combine(root, "samples"), "*.cs", SearchOption.AllDirectories)
            .Concat(Directory.GetFiles(Path.Combine(root, "tests"), "*.cs", SearchOption.AllDirectories))
            .Concat(Directory.GetFiles(Path.Combine(root, "src", "Feather", "NN"), "*.cs", SearchOption.AllDirectories))
            .Select(path => new { Path = path, Text = File.ReadAllText(path) })
            .ToArray();

        foreach (var source in sampleAndTestSources)
        {
            Assert.DoesNotMatch(@"trainer\.ADKernel|\.(Scratch|Tokens|Features|Labels)\b", source.Text);
        }

        var publicTrainerInternalsPattern = new Regex(@"public\s+.*\b(ADKernel|Scratch|Tokens|Features|Labels)\b", RegexOptions.Multiline);
        foreach (var source in nnSources)
        {
            Assert.DoesNotMatch(publicTrainerInternalsPattern, source);
            Assert.DoesNotContain("public GpuBuffer<float> Loss", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public GpuBuffer<int> Tokens", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public GpuBuffer<float> Features", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public GpuBuffer<float> Labels", source, StringComparison.Ordinal);
        }

        Assert.DoesNotMatch(@"public\s+.*PredictNext\(", sequenceModels);
        Assert.DoesNotMatch(@"public\s+.*Forward\(", sequenceModels);
        Assert.Contains("PredictNextHost", sequenceModels);
        Assert.Contains("ForwardHost", sequenceModels);
        Assert.Contains("RunHost", sequenceModels);
        Assert.Contains("public float LastLoss", sequenceModels);
        Assert.Contains("public DispatchPath LastDispatchPath", sequenceModels);
        Assert.Contains("public bool GradientsMaterialized", sequenceModels);
    }

    private static bool IsNativeReferenceFallbackTrait(CustomAttributeData attribute)
        => attribute.AttributeType == typeof(TraitAttribute)
            && attribute.ConstructorArguments.Count == 2
            && (string?)attribute.ConstructorArguments[0].Value == "Coverage"
            && (string?)attribute.ConstructorArguments[1].Value == "NativeReferenceFallback";

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Feather.slnx")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not find repository root.");
    }

    private static string SliceBetween(string text, string start, string end)
    {
        var startIndex = text.IndexOf(start, StringComparison.Ordinal);
        Assert.True(startIndex >= 0, $"Missing start marker: {start}");
        var endIndex = text.IndexOf(end, startIndex + start.Length, StringComparison.Ordinal);
        Assert.True(endIndex > startIndex, $"Missing end marker: {end}");
        return text[startIndex..endIndex];
    }
}
