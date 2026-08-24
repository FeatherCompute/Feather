using System.Reflection;
using System.Text;
using System.Text.Json;
using Feather.Assets;
using Feather.Assets.Graphics;
using Feather.Assets.Scenes;
using Feather.Math;

namespace Feather.Tests;

public sealed class StandardAssetContractTests
{
    private const string Hash = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    [Fact]
    public void FiveFoundationTypesExposeStableNominalCapabilityAndProductContracts()
    {
        AssertFoundation<TextureAsset>(
            "c2f0c619-d756-42f2-bb4b-a4ca48ab6dd2",
            TextureViewCapability.CapabilityId,
            "7a10cc53-241d-4206-9631-e19d870ff370",
            TextureAsset.Contract.Outputs.TextureView,
            "d80529aa-1c69-47f2-ae46-63a7cb1e0916",
            TextureViewOutput.ContractId,
            "59f5b9eb-20a0-47e1-a61f-aad9e0da2aee");
        AssertFoundation<MaterialAsset>(
            "293fd339-bf12-41dd-9f98-0519c9e17418",
            MaterialDomainCapability.CapabilityId,
            "6b2caa4f-cd3a-41c0-bf66-f9a4df88b821",
            MaterialAsset.Contract.Outputs.MaterialBinding,
            "0e6b623d-20be-4779-9fdf-7d4cfa662963",
            MaterialBindingOutput.ContractId,
            "6ff31203-b0ce-46b1-a1ef-5f06d74e247a");
        AssertFoundation<ModelAsset>(
            "8ade6d04-a60d-4a58-9ec6-33e039f3b6a0",
            ModelInstantiationCapability.CapabilityId,
            "b3bf0ffb-c538-4249-a30a-d7e2159f4778",
            ModelAsset.Contract.Outputs.ModelTemplate,
            "23996109-e2a2-4f17-91f6-790cab1bd4dc",
            ModelTemplateOutput.ContractId,
            "c0395c08-d676-46e5-a3a4-d9f52f5ca4ec");
        AssertFoundation<SceneAsset>(
            "b934179f-2772-4419-afbc-a321888ec2ea",
            SceneSnapshotCapability.CapabilityId,
            "5cd3ec1b-629a-46ef-9b8a-55f6e96102db",
            SceneAsset.Contract.Outputs.SceneSnapshot,
            "d25e6564-e47b-48c2-a1b8-700a42c9c7af",
            SceneSnapshotOutput.ContractId,
            "ecf5d49e-e75e-4fbf-be68-c0d07813f612");
        AssertFoundation<ActorAsset>(
            "09dfd6df-e3b0-4bc2-882a-f42faf6be488",
            ActorInstantiationCapability.CapabilityId,
            "00e8076d-166c-4850-bf32-375aaed78602",
            ActorAsset.Contract.Outputs.ActorTemplate,
            "4a209b60-c61b-441b-b0ba-87bf09671f86",
            ActorTemplateOutput.ContractId,
            "81835893-81e0-458d-80fb-681c23b56d27");
    }

    [Fact]
    public void FoundationDescriptorsAreBoundedTypedAndContainNoRuntimeHandleOrPath()
    {
        var bounds = new AssetBounds3D(new float3(-1, -2, -3), new float3(1, 2, 3));
        var actor = new ActorTemplateOutput(
            1,
            [new ActorComponentContractId(Guid.Parse("7f667b0e-fb83-456d-b2dc-8fc038ac5108"))],
            bounds,
            hasPose: true,
            hasAnimation: false,
            AssetContentHash.Parse(Hash));
        var model = new ModelTemplateOutput(1, bounds, true, true, true, AssetContentHash.Parse(Hash));

        Assert.Single(actor.ComponentContractIds);
        Assert.True(model.HasHierarchy);
        Assert.Throws<ArgumentException>(() => new ActorTemplateOutput(
            1,
            [
                new ActorComponentContractId(Guid.Parse("7f667b0e-fb83-456d-b2dc-8fc038ac5108")),
                new ActorComponentContractId(Guid.Parse("7f667b0e-fb83-456d-b2dc-8fc038ac5108")),
            ],
            null,
            false,
            false,
            AssetContentHash.Parse(Hash)));
        Type[] descriptors =
        [
            typeof(TextureViewOutput),
            typeof(MaterialBindingOutput),
            typeof(ModelTemplateOutput),
            typeof(SceneSnapshotOutput),
            typeof(ActorTemplateOutput),
        ];
        Assert.All(descriptors, descriptor => Assert.DoesNotContain(
            descriptor.GetProperties(),
            property => property.Name.Contains("Path", StringComparison.Ordinal) ||
                        property.Name.Contains("Handle", StringComparison.Ordinal) ||
                        property.PropertyType == typeof(nint) || property.PropertyType == typeof(ulong)));
    }

    [Fact]
    public void FeatherAssemblyPublishesRelativeClosedFoundationManifest()
    {
        AssemblyMetadataAttribute metadata = Assert.Single(
            typeof(Asset).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>(),
            attribute => attribute.Key == "Feather.AssetManifest");
        byte[] jsonBytes = Convert.FromBase64String(Assert.IsType<string>(metadata.Value));
        using JsonDocument document = JsonDocument.Parse(jsonBytes);
        JsonElement root = document.RootElement;
        string[] typeIds = root.GetProperty("assetTypes")
            .EnumerateArray()
            .Select(static type => type.GetProperty("typeId").GetString()!)
            .ToArray();
        string manifest = Encoding.UTF8.GetString(jsonBytes);

        Assert.Equal(5, typeIds.Length);
        Assert.Contains(TextureAsset.Contract.TypeId.ToString(), typeIds);
        Assert.Contains(MaterialAsset.Contract.TypeId.ToString(), typeIds);
        Assert.Contains(ModelAsset.Contract.TypeId.ToString(), typeIds);
        Assert.Contains(SceneAsset.Contract.TypeId.ToString(), typeIds);
        Assert.Contains(ActorAsset.Contract.TypeId.ToString(), typeIds);
        Assert.DoesNotContain("/Users/", manifest, StringComparison.Ordinal);
        Assert.DoesNotContain("\\Users\\", manifest, StringComparison.Ordinal);
        Assert.Contains("three-dimensional", manifest, StringComparison.Ordinal);
        Assert.Contains("never a learned or inference model", manifest, StringComparison.Ordinal);
    }

    private static void AssertFoundation<TAsset>(
        string expectedTypeId,
        AssetCapabilityId capabilityId,
        string expectedCapabilityId,
        AssetProductSlotId slotId,
        string expectedSlotId,
        Guid outputContractId,
        string expectedOutputContractId)
        where TAsset : Asset
    {
        Type type = typeof(TAsset);
        FeatherAssetTypeAttribute typeAttribute = Assert.Single(type.GetCustomAttributes<FeatherAssetTypeAttribute>());
        Assert.True(type.IsAbstract);
        Assert.Equal(expectedTypeId, typeAttribute.Guid);
        Assert.Equal(expectedCapabilityId, capabilityId.ToString());
        Assert.Equal(expectedSlotId, slotId.ToString());
        Assert.Equal(expectedOutputContractId, outputContractId.ToString("D"));
        Assert.DoesNotContain(type.GetProperties(), property =>
            property.PropertyType == typeof(string) || property.PropertyType == typeof(byte[]) ||
            property.PropertyType == typeof(nint) || property.PropertyType == typeof(ulong));
    }
}
