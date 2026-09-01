using Feather.Assets;

namespace Feather.Tests;

public sealed class AssetPreviewProgramContractTests
{
    [Fact]
    public void ProviderSelectsAStableTypedCSharpPreviewProgram()
    {
        var context = new RecordingPreviewContext();

        context.UseProgram<TestTexturePreviewProgram>();

        Assert.Equal(typeof(TestTexturePreviewProgram), context.ProgramType);
        var identity = Assert.Single(typeof(TestTexturePreviewProgram)
            .GetCustomAttributes(typeof(FeatherAssetPreviewProgramAttribute), false)
            .Cast<FeatherAssetPreviewProgramAttribute>());
        Assert.Equal("d1111111-1111-4111-8111-111111111111", identity.Guid);
        Assert.Equal("Texture Preview Demo", identity.Name);
    }

    private sealed class TestTextureAsset : Asset;

    [FeatherAssetPreviewProgram(
        "d1111111-1111-4111-8111-111111111111",
        Name = "Texture Preview Demo")]
    private sealed class TestTexturePreviewProgram : IAssetPreviewProgram<TestTextureAsset>
    {
        public void Render(AssetPreviewProgramContext<TestTextureAsset> context)
        {
            ArgumentNullException.ThrowIfNull(context);
        }
    }

    private sealed class RecordingPreviewContext : AssetPreviewContext<TestTextureAsset>
    {
        public Type? ProgramType { get; private set; }

        public override AssetProviderId ProviderId => AssetProviderId.Parse("d2222222-2222-4222-8222-222222222222");

        public override AssetContentHash ProviderImplementationHash =>
            AssetContentHash.Parse("sha256:" + new string('a', 64));

        public override long MaximumInputBytes => 1024;

        public override long MaximumOutputBytes => 1024;

        public override AssetRevisionSnapshot<TestTextureAsset> Revision => new(
            AssetId.Parse("d3333333-3333-4333-8333-333333333333"),
            AssetRevisionId.Parse("d4444444-4444-4444-8444-444444444444"),
            AssetContentHash.Parse("sha256:" + new string('b', 64)),
            new TestTextureAsset());

        public override string Profile => "thumbnail";

        public override string Target => "vulkan";

        protected override void SelectProgram(Type programType) => ProgramType = programType;
    }
}

