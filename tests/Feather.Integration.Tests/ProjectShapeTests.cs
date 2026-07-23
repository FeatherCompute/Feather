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
    public void NativeBuildLinksEasyGpuCore()
    {
        var root = FindRepositoryRoot();
        var cmakePath = Path.Combine(root, "native", "CMakeLists.txt");
        var text = File.ReadAllText(cmakePath);

        Assert.Contains("add_subdirectory(\"${CMAKE_CURRENT_SOURCE_DIR}/../EasyGPU\"", text);
        Assert.Contains("target_link_libraries(feather PRIVATE EasyGPU::EasyGPU)", text);
        Assert.Contains("EASYGPU_BUILD_TESTS OFF", text);
        Assert.Contains("EASYGPU_BUILD_EXAMPLES OFF", text);
    }

    [Fact]
    public void NativeReleaseBuildEnablesProductionSpirvOptimization()
    {
        var root = FindRepositoryRoot();
        var cmake = File.ReadAllText(Path.Combine(root, "native", "CMakeLists.txt"));
        var nativeBridge = File.ReadAllText(Path.Combine(root, "native", "feather_c_api.cpp"));
        var typedLowerer = File.ReadAllText(Path.Combine(root, "native", "feather_typed_ir_lowerer.cpp"));
        var ciWorkflow = File.ReadAllText(Path.Combine(root, ".github", "workflows", "ci.yml"));
        var releaseWorkflow = File.ReadAllText(Path.Combine(root, ".github", "workflows", "release.yml"));

        Assert.Contains("FEATHER_SHADER_OPTIMIZATION_LEVEL \"Ultra\"", cmake);
        Assert.Contains("context->SetOptimizationLevel(kShaderOptimizationLevel)", nativeBridge);
        Assert.Contains("forwardContext->SetOptimizationLevel(kShaderOptimizationLevel)", nativeBridge);
        Assert.Contains("vertex_shader_desc.optimizationLevel = kShaderOptimizationLevel", nativeBridge);
        Assert.Contains("fragment_shader_desc.optimizationLevel = kShaderOptimizationLevel", nativeBridge);
        Assert.Contains("shader_desc.optimizationLevel = kShaderOptimizationLevel", nativeBridge);
        Assert.Contains("kEnableFusedMultiplyAdd", nativeBridge);
        Assert.Contains("!kernel.auto_diff", nativeBridge);
        Assert.Contains("builder_.Intrinsic(\"fma\"", typedLowerer);
        Assert.Contains("EASYGPU_ENABLE_SPIRV_OPT=ON", ciWorkflow);
        Assert.Contains("EASYGPU_ENABLE_SPIRV_OPT=ON", releaseWorkflow);
        Assert.DoesNotContain("EASYGPU_ENABLE_SPIRV_OPT=OFF", ciWorkflow);
        Assert.DoesNotContain("EASYGPU_ENABLE_SPIRV_OPT=OFF", releaseWorkflow);
    }

    [Fact]
    public void EasyGpuModuleBarriersLowerThroughStructuralNode()
    {
        var root = FindRepositoryRoot();
        var modulePath = Path.Combine(root, "EasyGPU", "source", "IR", "Module.cpp");
        var text = File.ReadAllText(modulePath);
        var barrierCase = SliceBetween(
            text,
            "case Statement::Kind::Barrier:",
            "case Statement::Kind::SharedMemoryDecl:");
        var barrierLowerer = SliceBetween(
            text,
            "[[nodiscard]] bool LowerBarrierStatement",
            "[[nodiscard]] bool LowerLocalDeclaration");

        Assert.Contains("return LowerBarrierStatement(statement);", barrierCase);
        Assert.DoesNotContain("PushTranslatedCode", barrierCase);
        Assert.Contains("Node::BarrierNode", barrierLowerer);
        Assert.Contains("Builder::Builder::Get().Build(barrierNode, true);", barrierLowerer);
    }

    [Fact]
    public void EasyGpuModuleLocalDeclarationsLowerThroughStructuralNode()
    {
        var root = FindRepositoryRoot();
        var modulePath = Path.Combine(root, "EasyGPU", "source", "IR", "Module.cpp");
        var text = File.ReadAllText(modulePath);
        var localLowerer = SliceBetween(
            text,
            "[[nodiscard]] bool LowerLocalDeclaration",
            "[[nodiscard]] bool LowerStatementToNodes");

        Assert.Contains("Node::LocalVariableNode", localLowerer);
        Assert.Contains("Builder::Builder::Get().Build(declaration, true);", localLowerer);
        Assert.DoesNotContain("PushTranslatedCode", localLowerer);
    }

    [Fact]
    public void EasyGpuModuleTextureStoresLowerThroughStructuralNode()
    {
        var root = FindRepositoryRoot();
        var modulePath = Path.Combine(root, "EasyGPU", "source", "IR", "Module.cpp");
        var text = File.ReadAllText(modulePath);
        var textureStoreLowerer = SliceBetween(
            text,
            "[[nodiscard]] bool LowerTextureStore",
            "[[nodiscard]] bool LowerBarrierStatement");

        Assert.Contains("Node::TextureStoreNode", textureStoreLowerer);
        Assert.Contains("Builder::Builder::Get().Build(store, true);", textureStoreLowerer);
        Assert.DoesNotContain("PushTranslatedCode", textureStoreLowerer);
        Assert.DoesNotContain("imageStore", textureStoreLowerer);
    }

    [Fact]
    public void EasyGpuModuleTextureReadsAndSamplesLowerThroughStructuralNodes()
    {
        var root = FindRepositoryRoot();
        var modulePath = Path.Combine(root, "EasyGPU", "source", "IR", "Module.cpp");
        var text = File.ReadAllText(modulePath);
        var textureElementBuilder = SliceBetween(
            text,
            "[[nodiscard]] std::unique_ptr<Node::Node> BuildTextureElement",
            "[[nodiscard]] std::unique_ptr<Node::Node> BuildPushConstant");
        var textureSampleBuilder = SliceBetween(
            text,
            "[[nodiscard]] std::unique_ptr<Node::Node> BuildTextureSample",
            "[[nodiscard]] std::unique_ptr<Node::Node> BuildAtomic");

        Assert.Contains("Node::TextureLoadNode", textureElementBuilder);
        Assert.DoesNotContain("imageLoad", textureElementBuilder);
        Assert.Contains("Node::TextureSampleNode", textureSampleBuilder);
        Assert.DoesNotContain("texture(", textureSampleBuilder);
        Assert.DoesNotContain("textureLod", textureSampleBuilder);
    }

    [Fact]
    public void EasyGpuModuleSharedMemoryDeclarationsLowerThroughStructuralNode()
    {
        var root = FindRepositoryRoot();
        var modulePath = Path.Combine(root, "EasyGPU", "source", "IR", "Module.cpp");
        var text = File.ReadAllText(modulePath);
        var sharedMemoryLowerer = SliceBetween(
            text,
            "[[nodiscard]] bool LowerSharedMemoryDeclaration",
            "[[nodiscard]] bool LowerLocalDeclaration");

        Assert.Contains("Node::SharedMemoryNode", sharedMemoryLowerer);
        Assert.Contains("Builder::Builder::Get().Build(sharedMemory, true);", sharedMemoryLowerer);
        Assert.DoesNotContain("PushSharedMemoryDeclaration", sharedMemoryLowerer);
        Assert.DoesNotContain("PushTranslatedCode", sharedMemoryLowerer);
    }

    [Fact]
    public void EasyGpuModuleForLoopsLowerThroughStructuralNode()
    {
        var root = FindRepositoryRoot();
        var modulePath = Path.Combine(root, "EasyGPU", "source", "IR", "Module.cpp");
        var text = File.ReadAllText(modulePath);
        var forLowerer = SliceBetween(
            text,
            "[[nodiscard]] bool LowerForStatement",
            "[[nodiscard]] bool LowerWhileStatement");

        Assert.Contains("Node::ForNode", forLowerer);
        Assert.Contains("Builder::Builder::Get().Build(forNode, true);", forLowerer);
        Assert.Contains("BuildStatementNodes(statement.bodyBlock)", forLowerer);
        Assert.DoesNotContain("PushTranslatedCode", forLowerer);
        Assert.DoesNotContain("\"for (\"", forLowerer);
    }

    [Fact]
    public void EasyGpuModuleControlFlowBodiesLowerThroughStructuralNodes()
    {
        var root = FindRepositoryRoot();
        var modulePath = Path.Combine(root, "EasyGPU", "source", "IR", "Module.cpp");
        var text = File.ReadAllText(modulePath);
        var ifLowerer = SliceBetween(
            text,
            "[[nodiscard]] bool LowerIfStatement",
            "[[nodiscard]] bool LowerIfStatementToNodes");
        var whileLowerer = SliceBetween(
            text,
            "[[nodiscard]] bool LowerWhileStatement",
            "[[nodiscard]] bool LowerWhileStatementToNodes");
        var doWhileLowerer = SliceBetween(
            text,
            "[[nodiscard]] bool LowerDoWhileStatement",
            "[[nodiscard]] bool LowerDoWhileStatementToNodes");

        Assert.Contains("BuildStatementNodes(statement.thenBlock)", ifLowerer);
        Assert.Contains("BuildStatementNodes(statement.bodyBlock)", whileLowerer);
        Assert.Contains("BuildStatementNodes(statement.bodyBlock)", doWhileLowerer);
        Assert.DoesNotContain("CollectedCodeToNodes", ifLowerer);
        Assert.DoesNotContain("CodeCollectContext", ifLowerer);
        Assert.DoesNotContain("CollectedCodeToNodes", whileLowerer);
        Assert.DoesNotContain("CodeCollectContext", whileLowerer);
        Assert.DoesNotContain("CollectedCodeToNodes", doWhileLowerer);
        Assert.DoesNotContain("CodeCollectContext", doWhileLowerer);
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
