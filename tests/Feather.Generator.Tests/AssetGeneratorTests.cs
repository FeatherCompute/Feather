using System.Text.Json;
using Feather.Assets;
using Feather.Generators;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Feather.Generator.Tests;

public sealed class AssetGeneratorTests
{
    private const string CapabilityGuid = "04052f23-dc9c-4f0a-9770-ebd35bfddbbb";
    private const string OutputContractGuid = "81b8755c-a712-4c21-9e76-0e13a48eda43";
    private const string BaseTypeGuid = "2d0a7b51-4bf5-4ec2-89ea-2a184a073d0f";
    private const string TypeGuid = "878827ac-7fe1-4990-acad-554923b696c8";
    private const string ScaleGuid = "0228c70f-7456-416f-807d-f4cd4b96e859";
    private const string LabelGuid = "79574624-0838-4cef-8962-aab26ad1ea26";
    private const string ReferenceGuid = "5b2fd1cb-0a91-421a-8858-21f153547e3a";
    private const string SlotGuid = "32087aaa-22f8-4033-95f3-f86a4654614b";
    private const string ProviderGuid = "aa4c24ec-750a-4c1d-aab1-4fbb66d6e474";
    private const string ReferenceSocketGuid = "34fda284-f00f-4a18-8174-8ce93d353d67";
    private const string ProductInputSocketGuid = "62265a65-a9b4-401c-bf79-3dd70386ad8b";
    private const string ProductOutputSocketGuid = "86dc808c-1a63-4af9-a950-cdc477d80109";
    private const string TextureSocketGuid = "b1acfd3b-9521-47dd-a975-c23f0ca12c57";

    [Fact]
    public void GeneratorEmitsDeterministicAssetContractsProvidersAndCompanion()
    {
        var first = Generate(ValidSource, "Assets/GradientField.cs");
        var second = Generate(ValidSource, "Assets/GradientField.cs");

        Assert.DoesNotContain(first.Diagnostics, static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.DoesNotContain(first.Output.GetDiagnostics(), static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.Equal(first.Manifest, second.Manifest);

        using var document = JsonDocument.Parse(first.Manifest);
        var root = document.RootElement;
        Assert.Equal(1, root.GetProperty("schemaVersion").GetInt32());
        Assert.Equal("Feather.AssetAssemblyManifest", root.GetProperty("kind").GetString());
        Assert.Matches("^sha256:[0-9a-f]{64}$", root.GetProperty("buildId").GetString());

        var capability = Assert.Single(root.GetProperty("capabilityContracts").EnumerateArray());
        Assert.Equal(CapabilityGuid, capability.GetProperty("contractId").GetString());
        Assert.Equal("Scratch.FieldSampling", capability.GetProperty("typeName").GetString());
        var outputContract = Assert.Single(root.GetProperty("outputContracts").EnumerateArray());
        Assert.Equal(OutputContractGuid, outputContract.GetProperty("contractId").GetString());

        var types = root.GetProperty("assetTypes").EnumerateArray().ToArray();
        Assert.Equal(2, types.Length);
        var gradient = types.Single(type => type.GetProperty("typeId").GetString() == TypeGuid);
        Assert.Equal("Scratch.GradientFieldAsset", gradient.GetProperty("typeName").GetString());
        Assert.Equal(BaseTypeGuid, gradient.GetProperty("baseType").GetProperty("typeId").GetString());
        Assert.Equal(BaseTypeGuid, Assert.Single(gradient.GetProperty("ancestry").EnumerateArray()).GetString());
        Assert.Equal("Assets/GradientField.cs", gradient.GetProperty("source").GetProperty("path").GetString());

        var inputs = gradient.GetProperty("inputs").EnumerateArray()
            .ToDictionary(input => input.GetProperty("inputId").GetString()!);
        Assert.Equal("FLOAT", inputs[ScaleGuid].GetProperty("valueKind").GetString());
        Assert.Equal(0.01, inputs[ScaleGuid].GetProperty("step").GetDouble());
        Assert.Equal("TEXT", inputs[LabelGuid].GetProperty("valueKind").GetString());
        Assert.Equal(64, inputs[LabelGuid].GetProperty("maximumLength").GetInt32());
        Assert.Equal("ASSET_REFERENCE", inputs[ReferenceGuid].GetProperty("valueKind").GetString());
        Assert.Equal("Scratch.FieldAsset", inputs[ReferenceGuid].GetProperty("referencedAssetType").GetString());

        var capabilityUse = Assert.Single(gradient.GetProperty("capabilities").EnumerateArray());
        Assert.Equal(CapabilityGuid, capabilityUse.GetProperty("capabilityId").GetString());
        Assert.Equal(1, capabilityUse.GetProperty("minimumVersion").GetProperty("major").GetInt32());
        var slot = Assert.Single(gradient.GetProperty("productSlots").EnumerateArray());
        Assert.Equal(SlotGuid, slot.GetProperty("slotId").GetString());
        Assert.Equal(OutputContractGuid, slot.GetProperty("outputContractId").GetString());
        Assert.Equal(
            ["INPUT", "OUTPUT"],
            slot.GetProperty("passDirections").EnumerateArray().Select(static item => item.GetString()));

        var provider = Assert.Single(root.GetProperty("providers").EnumerateArray());
        Assert.Equal(ProviderGuid, provider.GetProperty("providerId").GetString());
        Assert.Equal("BUILD", provider.GetProperty("operation").GetString());
        Assert.Equal("ISOLATED_WORKER", provider.GetProperty("owner").GetString());
        Assert.Equal("Scratch.GradientFieldAsset", Assert.Single(provider.GetProperty("assetTypes").EnumerateArray()).GetString());

        using var passDocument = JsonDocument.Parse(first.PassManifest);
        var pass = Assert.Single(passDocument.RootElement.GetProperty("passes").EnumerateArray());
        var sockets = pass.GetProperty("inputs").EnumerateArray()
            .Concat(pass.GetProperty("outputs").EnumerateArray())
            .ToDictionary(socket => socket.GetProperty("socketGuid").GetString()!);
        var referenceSocket = sockets[ReferenceSocketGuid];
        Assert.Equal("ASSET_REFERENCE", referenceSocket.GetProperty("contractKind").GetString());
        Assert.Equal(
            TypeGuid,
            referenceSocket.GetProperty("assetContract").GetProperty("requiredTypeId").GetString());
        Assert.Equal(
            CapabilityGuid,
            Assert.Single(referenceSocket.GetProperty("assetContract").GetProperty("requiredCapabilities").EnumerateArray())
                .GetProperty("capabilityId").GetString());
        Assert.False(referenceSocket.GetProperty("assetContract").GetProperty("adapterRequired").GetBoolean());

        foreach (var socketGuid in new[] { ProductInputSocketGuid, ProductOutputSocketGuid })
        {
            var productSocket = sockets[socketGuid];
            Assert.Equal("ASSET_PRODUCT", productSocket.GetProperty("contractKind").GetString());
            Assert.Equal(SlotGuid, productSocket.GetProperty("assetContract").GetProperty("productSlotId").GetString());
            Assert.Equal(OutputContractGuid, productSocket.GetProperty("assetContract").GetProperty("outputContractId").GetString());
            Assert.True(productSocket.GetProperty("assetContract").GetProperty("adapterRequired").GetBoolean());
        }
        Assert.Equal("GPU_RESOURCE", sockets[TextureSocketGuid].GetProperty("contractKind").GetString());

        var companion = first.Output.SyntaxTrees.Single(tree =>
            tree.FilePath.EndsWith("Scratch_GradientFieldAsset.Feather.AssetContract.g.cs", StringComparison.Ordinal));
        var companionText = companion.ToString();
        Assert.Contains($"AssetTypeId.Parse(\"{TypeGuid}\")", companionText, StringComparison.Ordinal);
        Assert.Contains($"AssetInputId.Parse(\"{ScaleGuid}\")", companionText, StringComparison.Ordinal);
        Assert.Contains($"AssetProductSlotId.Parse(\"{SlotGuid}\")", companionText, StringComparison.Ordinal);
    }

    [Fact]
    public void StableIdsSurviveTypeSymbolAndSourceRename()
    {
        var before = Generate(ValidSource, "Assets/GradientField.cs");
        var renamed = Generate(
            ValidSource.Replace("GradientFieldAsset", "RenamedFieldAsset", StringComparison.Ordinal),
            "Moved/RenamedField.cs");

        using var beforeDocument = JsonDocument.Parse(before.Manifest);
        using var renamedDocument = JsonDocument.Parse(renamed.Manifest);
        var beforeType = TypeById(beforeDocument.RootElement, TypeGuid);
        var renamedType = TypeById(renamedDocument.RootElement, TypeGuid);

        Assert.Equal(TypeGuid, renamedType.GetProperty("typeId").GetString());
        Assert.Equal(
            beforeType.GetProperty("inputs").EnumerateArray().Select(static input => input.GetProperty("inputId").GetString()),
            renamedType.GetProperty("inputs").EnumerateArray().Select(static input => input.GetProperty("inputId").GetString()));
        Assert.Equal(
            beforeType.GetProperty("productSlots").EnumerateArray().Select(static output => output.GetProperty("slotId").GetString()),
            renamedType.GetProperty("productSlots").EnumerateArray().Select(static output => output.GetProperty("slotId").GetString()));
        Assert.NotEqual(
            beforeDocument.RootElement.GetProperty("buildId").GetString(),
            renamedDocument.RootElement.GetProperty("buildId").GetString());
    }

    [Fact]
    public void GeneratorRejectsInvalidIdentityMutabilityCapabilityOutputAndProviderContracts()
    {
        var result = Generate(
            """
            using System.Threading;
            using System.Threading.Tasks;
            using Feather.Assets;

            [FeatherAssetCapability("04052f23-dc9c-4f0a-9770-ebd35bfddbbb")]
            public sealed class ValidCapability : IAssetCapabilityContract { }

            public sealed class MissingCapability : IAssetCapabilityContract { }
            public sealed class MissingOutput : IAssetOutputContract { }

            [FeatherAssetType("878827AC-7FE1-4990-ACAD-554923B696C8")]
            public sealed partial class UppercaseIdentity : Asset { }

            [FeatherAssetType("878827ac-7fe1-4990-acad-554923b696c8")]
            [AssetCapability<MissingCapability>]
            [AssetOutput<MissingOutput>("32087aaa-22f8-4033-95f3-f86a4654614b", Symbol = "Bad Output")]
            public sealed partial class InvalidMembers : Asset
            {
                [AssetInput("0228c70f-7456-416f-807d-f4cd4b96e859")]
                public float Mutable { get; set; }

                [AssetInput("0228c70f-7456-416f-807d-f4cd4b96e859")]
                public float Duplicate { get; init; }
            }

            [FeatherAssetProvider(
                "aa4c24ec-750a-4c1d-aab1-4fbb66d6e474",
                AssetProviderOperation.Import)]
            public sealed class WrongProvider : IAssetBuilder<InvalidMembers>
            {
                public ValueTask BuildAsync(AssetBuildContext<InvalidMembers> context, CancellationToken cancellationToken)
                    => ValueTask.CompletedTask;
            }
            """,
            "Assets/Invalid.cs");

        var ids = result.Diagnostics.Select(static diagnostic => diagnostic.Id).ToHashSet(StringComparer.Ordinal);
        Assert.Contains("FSA001", ids);
        Assert.Contains("FSA010", ids);
        Assert.Contains("FSA011", ids);
        Assert.Contains("FSA020", ids);
        Assert.Contains("FSA030", ids);
        Assert.Contains("FSA040", ids);
        Assert.DoesNotContain(result.Output.SyntaxTrees, tree =>
            tree.FilePath.EndsWith("InvalidMembers.Feather.AssetContract.g.cs", StringComparison.Ordinal));
    }

    [Fact]
    public void GeneratorRejectsPreparedProductSocketWhoseBindingDoesNotMatchDeclaredSlot()
    {
        var result = Generate(
            """
            using Feather.Assets;
            using Feather.RenderGraph;

            [FeatherAssetOutputContract("81b8755c-a712-4c21-9e76-0e13a48eda43")]
            public sealed class Product : IAssetOutputContract { }

            [FeatherAssetType("878827ac-7fe1-4990-acad-554923b696c8")]
            [AssetOutput<Product>(
                "32087aaa-22f8-4033-95f3-f86a4654614b",
                Symbol = "Product")]
            public sealed partial class ProductAsset : Asset { }

            [FeatherPass("c9176c1c-54f0-4d5d-aa0f-3a1e35846b67")]
            public sealed class InvalidProductPass : IComputePass
            {
                [Input("62265a65-a9b4-401c-bf79-3dd70386ad8b")]
                [AssetProductBinding(
                    typeof(ProductAsset),
                    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")]
                public AssetOutputHandle<Product> Product { get; init; }

                public void Execute(RenderContext context) { }
            }
            """,
            "Assets/InvalidProductPass.cs");

        Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Id == "FSA020");
        using var manifest = JsonDocument.Parse(result.PassManifest);
        Assert.Empty(manifest.RootElement.GetProperty("passes").EnumerateArray());
    }

    private static GeneratorResult Generate(string source, string sourcePath)
    {
        var compilation = CreateCompilation(source, sourcePath);
        var driver = CSharpGeneratorDriver.Create(new FeatherGenerator());
        driver.RunGeneratorsAndUpdateCompilation(
            compilation,
            out var outputCompilation,
            out var diagnostics);
        var generated = outputCompilation.SyntaxTrees.SingleOrDefault(tree =>
            tree.FilePath.EndsWith("Feather.AssetManifest.g.cs", StringComparison.Ordinal));
        var manifest = generated is null ? string.Empty : ReadManifest(outputCompilation, generated);
        var passGenerated = outputCompilation.SyntaxTrees.SingleOrDefault(tree =>
            tree.FilePath.EndsWith("Feather.PassManifest.g.cs", StringComparison.Ordinal));
        var passManifest = passGenerated is null
            ? string.Empty
            : ReadManifest(outputCompilation, passGenerated);
        return new GeneratorResult(outputCompilation, diagnostics, manifest, passManifest);
    }

    private static string ReadManifest(Compilation compilation, SyntaxTree generated)
    {
        var variable = generated.GetRoot().DescendantNodes().OfType<VariableDeclaratorSyntax>()
            .Single(item => item.Identifier.ValueText == "Json");
        var field = Assert.IsAssignableFrom<IFieldSymbol>(
            compilation.GetSemanticModel(generated).GetDeclaredSymbol(variable));
        return Assert.IsType<string>(field.ConstantValue);
    }

    private static JsonElement TypeById(JsonElement root, string typeId)
        => root.GetProperty("assetTypes").EnumerateArray()
            .Single(type => type.GetProperty("typeId").GetString() == typeId);

    private static CSharpCompilation CreateCompilation(string source, string sourcePath)
    {
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(static assembly => !assembly.IsDynamic && !string.IsNullOrWhiteSpace(assembly.Location))
            .Select(static assembly => MetadataReference.CreateFromFile(assembly.Location))
            .Append(MetadataReference.CreateFromFile(typeof(Asset).Assembly.Location))
            .Distinct(MetadataReferenceComparer.Instance);
        return CSharpCompilation.Create(
            "AssetScratch",
            [CSharpSyntaxTree.ParseText(source, path: sourcePath)],
            references,
            new CSharpCompilationOptions(
                OutputKind.DynamicallyLinkedLibrary,
                nullableContextOptions: NullableContextOptions.Enable));
    }

    private sealed record GeneratorResult(
        Compilation Output,
        IReadOnlyList<Diagnostic> Diagnostics,
        string Manifest,
        string PassManifest);

    private sealed class MetadataReferenceComparer : IEqualityComparer<MetadataReference>
    {
        public static MetadataReferenceComparer Instance { get; } = new();

        public bool Equals(MetadataReference? x, MetadataReference? y)
            => StringComparer.OrdinalIgnoreCase.Equals(x?.Display, y?.Display);

        public int GetHashCode(MetadataReference value)
            => StringComparer.OrdinalIgnoreCase.GetHashCode(value.Display ?? string.Empty);
    }

    private static string ValidSource => $$"""
        using System.Threading;
        using System.Threading.Tasks;
        using Feather.Assets;
        using Feather.RenderGraph;

        namespace Scratch;

        [FeatherAssetCapability(
            "{{CapabilityGuid}}",
            Name = "Field Sampling",
            ContractMajor = 1,
            ContractMinor = 2)]
        public sealed class FieldSampling : IAssetCapabilityContract { }

        [FeatherAssetOutputContract(
            "{{OutputContractGuid}}",
            Name = "Dense Field",
            ContractMajor = 1)]
        public sealed class DenseFieldOutput : IAssetOutputContract { }

        [FeatherAssetType(
            "{{BaseTypeGuid}}",
            Name = "Field",
            Abstract = true)]
        public abstract partial class FieldAsset : Asset { }

        [FeatherAssetType(
            "{{TypeGuid}}",
            Name = "Gradient Field",
            Description = "A deliberately nontraditional field Asset",
            PayloadSchemaVersion = 2)]
        [AssetCapability<FieldSampling>(MinimumMajor = 1, MinimumMinor = 1)]
        [AssetOutput<DenseFieldOutput>(
            "{{SlotGuid}}",
            Symbol = "DenseField",
            Name = "Dense Field",
            PassDirections = AssetPassDirections.Input | AssetPassDirections.Output)]
        public sealed partial class GradientFieldAsset : FieldAsset
        {
            [AssetInput(
                "{{ScaleGuid}}",
                Name = "Scale",
                Min = 0,
                Max = 8,
                Step = 0.01,
                Role = AssetInputRole.Evaluation | AssetInputRole.Runtime,
                ChangeImpact = AssetChangeImpact.RuntimeCandidate)]
            public float Scale { get; init; } = 1;

            [AssetInput(
                "{{LabelGuid}}",
                Name = "Label",
                MaxLength = 64,
                ChangeImpact = AssetChangeImpact.MetadataOnly)]
            public string Label { get; init; } = string.Empty;

            [AssetInput(
                "{{ReferenceGuid}}",
                Name = "Source",
                Required = false,
                Role = AssetInputRole.Evaluation | AssetInputRole.Preview)]
            public AssetRef<FieldAsset> Source { get; init; }
        }

        [FeatherAssetProvider(
            "{{ProviderGuid}}",
            AssetProviderOperation.Build,
            Name = "Gradient Field Builder",
            Owner = AssetProviderOwner.IsolatedWorker)]
        public sealed class GradientFieldBuilder : IAssetBuilder<GradientFieldAsset>
        {
            public ValueTask BuildAsync(
                AssetBuildContext<GradientFieldAsset> context,
                CancellationToken cancellationToken)
                => ValueTask.CompletedTask;
        }

        [FeatherPass("c9176c1c-54f0-4d5d-aa0f-3a1e35846b67", Name = "Gradient Field Consumer")]
        public sealed class GradientFieldConsumer : IComputePass
        {
            [Input("{{ReferenceSocketGuid}}", Name = "Field")]
            public AssetRef<GradientFieldAsset> Field { get; init; }

            [Input("{{ProductInputSocketGuid}}", Name = "Prepared Field")]
            [AssetProductBinding(
                typeof(GradientFieldAsset),
                "{{SlotGuid}}")] // exact stable product slot
            public AssetOutputHandle<DenseFieldOutput> PreparedField { get; init; }

            [Output("{{ProductOutputSocketGuid}}", Name = "Published Field")]
            [AssetProductBinding(
                typeof(GradientFieldAsset),
                "{{SlotGuid}}")] // exact stable product slot
            public AssetOutputHandle<DenseFieldOutput> PublishedField { get; init; }

            [Input("{{TextureSocketGuid}}", Name = "Texture")]
            public TextureHandle Texture { get; init; }

            public void Execute(RenderContext context) { }
        }
        """;
}
