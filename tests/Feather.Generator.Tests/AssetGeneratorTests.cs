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
    private const string TintGuid = "8f0cd771-3c7f-47d8-a3d8-b51f5325128b";
    private const string ReferenceGuid = "5b2fd1cb-0a91-421a-8858-21f153547e3a";
    private const string SlotGuid = "32087aaa-22f8-4033-95f3-f86a4654614b";
    private const string ProviderGuid = "aa4c24ec-750a-4c1d-aab1-4fbb66d6e474";
    private const string ReferenceSocketGuid = "34fda284-f00f-4a18-8174-8ce93d353d67";
    private const string ProductInputSocketGuid = "62265a65-a9b4-401c-bf79-3dd70386ad8b";
    private const string ProductOutputSocketGuid = "86dc808c-1a63-4af9-a950-cdc477d80109";
    private const string TextureSocketGuid = "b1acfd3b-9521-47dd-a975-c23f0ca12c57";
    private const string DataTypeGuid = "de71ba15-fdbc-4ea1-9dcf-5e3551bb6985";
    private const string DataSocketGuid = "ec52ae35-1ef1-47a0-983b-6a8506f596da";
    private const string DataLayoutHash = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    [Fact]
    public void GeneratorPublishesCSharpAuthoredDataTypeAndGpuStructLayoutIdentity()
    {
        const string dataTypeId = "70ed1481-f8a5-4abd-a835-4f619a8de2c7";
        const string resourceId = "af0c4383-28df-44f5-884a-51820e69212e";
        var result = Generate(
            $$"""
            using Feather;
            using Feather.Math;
            using Feather.RenderGraph;

            namespace Scratch;

            [GpuStruct]
            public partial struct ProbeSample
            {
                public float3 Position;
                public float Weight;
            }

            [FeatherDataType("{{dataTypeId}}", Name = "Probe Field")]
            public sealed class ProbeFieldData
            {
                [DataResource(
                    "{{resourceId}}",
                    Name = "Probes",
                    Access = DataAccess.ReadWrite,
                    Creation = DataCreation.BeforeGraph,
                    Update = DataUpdate.PassMutated,
                    Lifetime = DataResourceLifetime.Workspace,
                    ElementCount = 64,
                    MaximumBytes = 1024)]
                public DataBuffer<ProbeSample> Probes;
            }
            """,
            "Data/Types/ProbeFieldData.cs");

        Assert.DoesNotContain(
            result.Diagnostics,
            static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.DoesNotContain(
            result.Output.GetDiagnostics(),
            static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        using var manifest = JsonDocument.Parse(result.DataManifest);
        Assert.Equal("Feather.DataAssemblyManifest", manifest.RootElement.GetProperty("kind").GetString());
        var type = Assert.Single(manifest.RootElement.GetProperty("types").EnumerateArray());
        Assert.Equal("Data/Types/ProbeFieldData.cs", type.GetProperty("sourcePath").GetString());
        var document = type.GetProperty("document");
        Assert.Equal(dataTypeId, document.GetProperty("typeId").GetString());
        Assert.Matches("^sha256:[0-9a-f]{64}$", document.GetProperty("manifestHash").GetString());
        Assert.Matches("^sha256:[0-9a-f]{64}$", document.GetProperty("layoutAbiHash").GetString());
        var resource = Assert.Single(document.GetProperty("resources").EnumerateArray());
        Assert.Equal(resourceId, resource.GetProperty("resourceId").GetString());
        Assert.Equal("BUFFER", resource.GetProperty("kind").GetString());
        Assert.Equal("Scratch.ProbeSample", resource.GetProperty("elementType").GetString());
        Assert.Matches("^sha256:[0-9a-f]{64}$", resource.GetProperty("elementLayoutAbiHash").GetString());
        Assert.Equal(16, resource.GetProperty("elementStrideBytes").GetInt32());
        Assert.Equal(16, resource.GetProperty("elementAlignmentBytes").GetInt32());
        Assert.Equal(64, resource.GetProperty("elementCount").GetInt64());
        Assert.Equal(1024, resource.GetProperty("maximumBytes").GetInt64());
    }

    [Fact]
    public void GeneratorRejectsInvalidDataIdentityAndUnboundedResource()
    {
        var result = Generate(
            """
            using Feather.RenderGraph;

            [FeatherDataType("70ED1481-F8A5-4ABD-A835-4F619A8DE2C7")]
            public sealed class BrokenData
            {
                [DataResource("not-a-guid")]
                public DataBuffer<float> Values;
            }
            """,
            "Data/Types/BrokenData.cs");

        var ids = result.Diagnostics.Select(static diagnostic => diagnostic.Id).ToHashSet();
        Assert.Contains("FSD002", ids);
        Assert.Empty(result.DataManifest);
    }

    [Fact]
    public void PassManifestKeepsDataInstanceAsOneExactTypedObject()
    {
        var result = Generate(
            $$"""
            using Feather.RenderGraph;

            [FeatherPass("527bbab3-436a-4c73-9e5e-0de711ad1d3c")]
            public sealed class ProbeLightingPass : IComputePass
            {
                [Input("{{DataSocketGuid}}", Name = "Probe GI")]
                [DataBinding("{{DataTypeGuid}}", "{{DataLayoutHash}}", ContractMajor = 1)]
                public DataHandle Probes { get; init; }

                public void Execute(RenderContext context) { }
            }
            """,
            "Passes/ProbeLightingPass.cs");

        Assert.DoesNotContain(
            result.Diagnostics,
            static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        using var manifest = JsonDocument.Parse(result.PassManifest);
        JsonElement input = Assert.Single(
            Assert.Single(manifest.RootElement.GetProperty("passes").EnumerateArray())
                .GetProperty("inputs").EnumerateArray());
        Assert.Equal("Data", input.GetProperty("resourceKind").GetString());
        Assert.Equal("DATA_INSTANCE", input.GetProperty("contractKind").GetString());
        Assert.Equal(
            DataTypeGuid,
            input.GetProperty("dataContract").GetProperty("requiredTypeId").GetString());
        Assert.Equal(
            DataLayoutHash,
            input.GetProperty("dataContract").GetProperty("layoutAbiHash").GetString());
    }

    [Fact]
    public void DataHandleWithoutAnExactBindingIsRejected()
    {
        var result = Generate(
            $$"""
            using Feather.RenderGraph;

            [FeatherPass("527bbab3-436a-4c73-9e5e-0de711ad1d3c")]
            public sealed class InvalidDataPass : IComputePass
            {
                [Input("{{DataSocketGuid}}")] public DataHandle Data { get; init; }
                public void Execute(RenderContext context) { }
            }
            """,
            "Passes/InvalidDataPass.cs");

        Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Id == "FSD001");
    }

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
        Assert.Equal(2, root.GetProperty("schemaVersion").GetInt32());
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
        Assert.Equal(1, inputs[ScaleGuid].GetProperty("defaultValue").GetDouble());
        Assert.Equal("TEXT", inputs[LabelGuid].GetProperty("valueKind").GetString());
        Assert.Equal(64, inputs[LabelGuid].GetProperty("maximumLength").GetInt32());
        Assert.Equal(string.Empty, inputs[LabelGuid].GetProperty("defaultValue").GetString());
        Assert.Equal(
            [0.2, 0.4, 0.6],
            inputs[TintGuid].GetProperty("defaultValue").EnumerateArray()
                .Select(static component => component.GetDouble()).ToArray());
        Assert.Equal("ASSET_REFERENCE", inputs[ReferenceGuid].GetProperty("valueKind").GetString());
        Assert.Equal("Scratch.FieldAsset", inputs[ReferenceGuid].GetProperty("referencedAssetType").GetString());
        Assert.Equal(BaseTypeGuid, inputs[ReferenceGuid].GetProperty("referencedAssetTypeId").GetString());

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
    public void DerivedAssetMayUseCapabilityAndOutputContractsFromReferencedFeatherAssembly()
    {
        var result = Generate(
            """
            using Feather.Assets;
            using Feather.Assets.Graphics;
            using Feather.Assets.Scenes;

            namespace Scratch;

            [FeatherAssetType("f52f813d-6f74-4fa6-9b76-3cb9554264a2", Name = "Imported Scene Model")]
            [AssetCapability<SceneSnapshotCapability>]
            [AssetOutput<SceneSnapshotOutput>(
                "d25e6564-e47b-48c2-a1b8-700a42c9c7af",
                Symbol = "SceneSnapshot",
                PassDirections = AssetPassDirections.Input)]
            public sealed partial class ImportedSceneModelAsset : ModelAsset;
            """,
            "Assets/ImportedSceneModelAsset.cs");

        Assert.DoesNotContain(
            result.Diagnostics,
            static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.DoesNotContain(
            result.Output.GetDiagnostics(),
            static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        using var manifest = JsonDocument.Parse(result.Manifest);
        Assert.Empty(manifest.RootElement.GetProperty("capabilityContracts").EnumerateArray());
        Assert.Empty(manifest.RootElement.GetProperty("outputContracts").EnumerateArray());
        JsonElement type = Assert.Single(manifest.RootElement.GetProperty("assetTypes").EnumerateArray());
        JsonElement capability = Assert.Single(type.GetProperty("capabilities").EnumerateArray());
        Assert.Equal(
            "5cd3ec1b-629a-46ef-9b8a-55f6e96102db",
            capability.GetProperty("capabilityId").GetString());
        JsonElement output = Assert.Single(type.GetProperty("productSlots").EnumerateArray());
        Assert.Equal(
            "ecf5d49e-e75e-4fbf-be68-c0d07813f612",
            output.GetProperty("outputContractId").GetString());
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
    public void PreparedProductSocketAcceptsAProductInheritedByAProjectDefinedAssetType()
    {
        const string derivedTypeGuid = "7c773436-0954-4937-afdf-150e827dc381";
        var result = Generate(
            $$"""
            using Feather.Assets;
            using Feather.RenderGraph;

            [FeatherAssetOutputContract("{{OutputContractGuid}}")]
            public sealed class Product : IAssetOutputContract { }

            [FeatherAssetType("{{BaseTypeGuid}}", Abstract = true)]
            [AssetOutput<Product>("{{SlotGuid}}", Symbol = "Product")]
            public abstract partial class BaseProductAsset : Asset { }

            [FeatherAssetType("{{derivedTypeGuid}}")]
            public sealed partial class ProjectProductAsset : BaseProductAsset { }

            [FeatherPass("c9176c1c-54f0-4d5d-aa0f-3a1e35846b67")]
            public sealed class DerivedProductPass : IComputePass
            {
                [Input("{{ProductInputSocketGuid}}")]
                [AssetProductBinding(typeof(ProjectProductAsset), "{{SlotGuid}}")]
                public AssetOutputHandle<Product> Product { get; init; }

                public void Execute(RenderContext context) { }
            }
            """,
            "Assets/DerivedProductPass.cs");

        Assert.DoesNotContain(
            result.Diagnostics,
            static diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        using var passDocument = JsonDocument.Parse(result.PassManifest);
        var pass = Assert.Single(passDocument.RootElement.GetProperty("passes").EnumerateArray());
        var input = Assert.Single(pass.GetProperty("inputs").EnumerateArray());
        Assert.Equal("ASSET_PRODUCT", input.GetProperty("contractKind").GetString());
        Assert.Equal(
            derivedTypeGuid,
            input.GetProperty("assetContract").GetProperty("requiredTypeId").GetString());
        Assert.Equal(
            SlotGuid,
            input.GetProperty("assetContract").GetProperty("productSlotId").GetString());
        Assert.Equal(
            OutputContractGuid,
            input.GetProperty("assetContract").GetProperty("outputContractId").GetString());
    }

    [Fact]
    public void StandardFoundationTypesAreExtensibleThroughReferencedPublicContracts()
    {
        var result = Generate(
            """
            using Feather.Assets;
            using Feather.Assets.Graphics;
            using Feather.Assets.Scenes;

            [FeatherAssetType("11111111-1111-4111-8111-111111111111", Name = "Custom Texture")]
            public sealed partial class CustomTexture : TextureAsset { }

            [FeatherAssetType("22222222-2222-4222-8222-222222222222", Name = "Custom Material")]
            public sealed partial class CustomMaterial : MaterialAsset { }

            [FeatherAssetType("33333333-3333-4333-8333-333333333333", Name = "Custom 3D Model")]
            public sealed partial class CustomModel : ModelAsset { }

            [FeatherAssetType("44444444-4444-4444-8444-444444444444", Name = "Custom Scene")]
            public sealed partial class CustomScene : SceneAsset { }

            [FeatherAssetType("55555555-5555-4555-8555-555555555555", Name = "Custom Actor")]
            public sealed partial class CustomActor : ActorAsset { }
            """,
            "Assets/StandardExtensions.cs");

        Assert.DoesNotContain(result.Diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        using var document = JsonDocument.Parse(result.Manifest);
        var expectedBases = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["11111111-1111-4111-8111-111111111111"] = "c2f0c619-d756-42f2-bb4b-a4ca48ab6dd2",
            ["22222222-2222-4222-8222-222222222222"] = "293fd339-bf12-41dd-9f98-0519c9e17418",
            ["33333333-3333-4333-8333-333333333333"] = "8ade6d04-a60d-4a58-9ec6-33e039f3b6a0",
            ["44444444-4444-4444-8444-444444444444"] = "b934179f-2772-4419-afbc-a321888ec2ea",
            ["55555555-5555-4555-8555-555555555555"] = "09dfd6df-e3b0-4bc2-882a-f42faf6be488",
        };
        JsonElement[] types = document.RootElement.GetProperty("assetTypes").EnumerateArray().ToArray();

        Assert.Equal(5, types.Length);
        foreach ((string typeId, string baseTypeId) in expectedBases)
        {
            JsonElement type = types.Single(candidate => candidate.GetProperty("typeId").GetString() == typeId);
            Assert.Equal(baseTypeId, type.GetProperty("baseType").GetProperty("typeId").GetString());
            Assert.Equal(baseTypeId, Assert.Single(type.GetProperty("ancestry").EnumerateArray()).GetString());
        }
    }

    [Fact]
    public void ManifestSealsEveryDeclarationToItsExactUtf8SourceText()
    {
        const string sourcePath = "Assets/GradientField.cs";
        GeneratorResult result = Generate(ValidSource, sourcePath);
        string expectedHash = "sha256:" + Convert.ToHexString(
            System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(ValidSource)))
            .ToLowerInvariant();
        using JsonDocument document = JsonDocument.Parse(result.Manifest);

        JsonElement[] declarations =
        [
            .. document.RootElement.GetProperty("capabilityContracts").EnumerateArray(),
            .. document.RootElement.GetProperty("outputContracts").EnumerateArray(),
            .. document.RootElement.GetProperty("providers").EnumerateArray(),
            .. document.RootElement.GetProperty("assetTypes").EnumerateArray(),
        ];

        Assert.NotEmpty(declarations);
        Assert.All(declarations, declaration =>
        {
            JsonElement source = declaration.GetProperty("source");
            Assert.Equal(sourcePath, source.GetProperty("path").GetString());
            Assert.Equal(expectedHash, source.GetProperty("documentHash").GetString());
        });
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
        var dataGenerated = outputCompilation.SyntaxTrees.SingleOrDefault(tree =>
            tree.FilePath.EndsWith("Feather.DataManifest.g.cs", StringComparison.Ordinal));
        var dataManifest = dataGenerated is null
            ? string.Empty
            : ReadManifest(outputCompilation, dataGenerated);
        return new GeneratorResult(
            outputCompilation,
            diagnostics,
            manifest,
            passManifest,
            dataManifest);
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
        string PassManifest,
        string DataManifest);

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
        using Feather.Math;
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

            [AssetInput("{{TintGuid}}", Name = "Tint", Min = 0, Max = 1)]
            public float3 Tint { get; init; } = new(0.2f, 0.4f, 0.6f);

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
