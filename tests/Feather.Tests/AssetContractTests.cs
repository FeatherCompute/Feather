using Feather.Assets;
using Feather.RenderGraph;

namespace Feather.Tests;

public sealed class AssetContractTests
{
    private const string AssetGuid = "79d92227-abee-484b-8640-cb1ae9ec6cb5";
    private const string RevisionGuid = "17ce0701-408e-4775-aa6a-715935af9151";
    private const string SlotGuid = "32087aaa-22f8-4033-95f3-f86a4654614b";
    private const string Hash = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    [Fact]
    public void StableIdsRequireCanonicalLowercaseNonEmptyUuids()
    {
        var id = AssetId.Parse(AssetGuid);

        Assert.Equal(AssetGuid, id.ToString());
        Assert.Equal(Guid.Parse(AssetGuid), id.Value);
        Assert.True(AssetId.TryParse(AssetGuid, out var parsed));
        Assert.Equal(id, parsed);
        Assert.False(AssetId.TryParse(AssetGuid.ToUpperInvariant(), out _));
        Assert.False(AssetId.TryParse(Guid.Empty.ToString("D"), out _));
        Assert.False(AssetId.TryParse("not-an-id", out _));
        Assert.Throws<ArgumentException>(() => new AssetId(Guid.Empty));
        Assert.Throws<FormatException>(() => AssetTypeId.Parse(AssetGuid.ToUpperInvariant()));
        Assert.Throws<InvalidOperationException>(() => default(AssetProviderId).ToString());
    }

    [Fact]
    public void ContentHashKeepsAlgorithmAndIntegrityBytesCanonical()
    {
        var hash = AssetContentHash.Parse(Hash);

        Assert.Equal("sha256", hash.Algorithm);
        Assert.Equal(Hash[7..], hash.Hex);
        Assert.Equal(Hash, hash.ToString());
        Assert.True(AssetContentHash.TryParse(Hash, out var parsed));
        Assert.Equal(hash, parsed);
        Assert.False(AssetContentHash.TryParse(Hash.ToUpperInvariant(), out _));
        Assert.False(AssetContentHash.TryParse("sha512:" + Hash[7..], out _));
        Assert.Throws<ArgumentException>(() => new AssetContentHash("sha256", "00"));
    }

    [Fact]
    public void RevisionSelectorsHaveExactlyFollowAndCompletePinnedShapes()
    {
        var follow = AssetRevisionSelector.FollowCompatibleCurrent;
        var pinned = AssetRevisionSelector.PinnedExact(
            AssetRevisionId.Parse(RevisionGuid),
            AssetContentHash.Parse(Hash));

        Assert.Equal(AssetRevisionPolicy.FollowCompatibleCurrent, follow.Policy);
        Assert.Null(follow.RevisionId);
        Assert.Null(follow.RevisionManifestHash);
        Assert.Equal(AssetRevisionPolicy.PinnedExact, pinned.Policy);
        Assert.Equal(RevisionGuid, pinned.RevisionId?.ToString());
        Assert.Equal(Hash, pinned.RevisionManifestHash?.ToString());
        Assert.Throws<ArgumentException>(() =>
            AssetRevisionSelector.PinnedExact(default, AssetContentHash.Parse(Hash)));
        Assert.Throws<InvalidOperationException>(() =>
            AssetRevisionSelector.PinnedExact(AssetRevisionId.Parse(RevisionGuid), default));
    }

    [Fact]
    public void LogicalReferencesCarryNoPathBytesOrRuntimeHandle()
    {
        var assetId = AssetId.Parse(AssetGuid);
        var reference = AssetRef<TestAsset>.Pin(
            assetId,
            AssetRevisionId.Parse(RevisionGuid),
            AssetContentHash.Parse(Hash));
        var output = new AssetOutputRef<TestAsset, TestOutput>(
            reference,
            AssetProductSlotId.Parse(SlotGuid));

        Assert.Equal(assetId, reference.AssetId);
        Assert.Equal(AssetRevisionPolicy.PinnedExact, reference.Revision.Policy);
        Assert.Equal(reference, output.Asset);
        Assert.Equal(SlotGuid, output.ProductSlotId.ToString());
        Assert.DoesNotContain(
            typeof(AssetRef<TestAsset>).GetProperties(),
            property => property.PropertyType == typeof(string) || property.PropertyType == typeof(byte[]));
        Assert.DoesNotContain(
            typeof(AssetOutputRef<TestAsset, TestOutput>).GetProperties(),
            property => property.PropertyType == typeof(ulong) || property.PropertyType == typeof(nint));
    }

    [Fact]
    public void AuthoringAnnotationsExposeOnlyStableContractMetadata()
    {
        var type = new FeatherAssetTypeAttribute(AssetGuid)
        {
            Name = "Gradient Field",
            ContractMajor = 2,
            ContractMinor = 1,
            PayloadSchemaVersion = 3,
        };
        var input = new AssetInputAttribute(RevisionGuid)
        {
            Required = false,
            Role = AssetInputRole.Evaluation | AssetInputRole.Preview,
            ChangeImpact = AssetChangeImpact.RuntimeCandidate,
            Min = -1,
            Max = 1,
            MaxItems = 64,
        };
        var provider = new FeatherAssetProviderAttribute(
            SlotGuid,
            AssetProviderOperation.Build)
        {
            Owner = AssetProviderOwner.IsolatedWorker,
            Determinism = AssetProviderDeterminism.Deterministic,
        };

        Assert.Equal(AssetGuid, type.Guid);
        Assert.Equal((ushort)2, type.ContractMajor);
        Assert.Equal(AssetInputRole.Evaluation | AssetInputRole.Preview, input.Role);
        Assert.Equal(64, input.MaxItems);
        Assert.Equal(AssetProviderOperation.Build, provider.Operation);
        Assert.Equal(AssetProviderOwner.IsolatedWorker, provider.Owner);
        Assert.DoesNotContain(
            typeof(AssetProviderContext).GetProperties(),
            property => property.Name.Contains("Service", StringComparison.Ordinal) ||
                        property.Name.Contains("Path", StringComparison.Ordinal) ||
                        property.Name.Contains("Handle", StringComparison.Ordinal));
    }

    [Fact]
    public void PreparedOutputHandleExposesNoPublicTokenOrConstructor()
    {
        var handleType = typeof(AssetOutputHandle<TestOutput>);

        Assert.Empty(handleType.GetConstructors());
        Assert.DoesNotContain(
            handleType.GetProperties(),
            property => property.PropertyType is { } type &&
                        (type == typeof(ulong) || type == typeof(nint) || type == typeof(string)));
        var binding = new AssetProductBindingAttribute(typeof(TestAsset), SlotGuid);
        Assert.Equal(typeof(TestAsset), binding.AssetType);
        Assert.Equal(SlotGuid, binding.ProductSlotGuid);
    }

    private sealed class TestAsset : Asset;

    private sealed class TestOutput : IAssetOutputContract;
}
