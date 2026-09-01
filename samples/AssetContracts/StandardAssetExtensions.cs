using Feather.Assets;
using Feather.Assets.Graphics;
using Feather.Assets.Scenes;

namespace AssetContracts;

// These two identities are consumed by Feather Studio's real lossless PNG and GLB import
// providers. The sample intentionally declares no file paths: source artifacts and revisions are
// owned by Asset service rather than embedded in the authoring payload.
[FeatherAssetType(
    "cad76d37-30f9-4483-97ba-3bc7691aef1a",
    Name = "PNG Texture",
    Description = "A losslessly imported PNG that realises the standard Texture View contract")]
public sealed partial class PngTextureAsset : TextureAsset;

[FeatherAssetType(
    "9b338c83-5397-46f2-a157-1e6e76adbb52",
    Name = "glTF Scene",
    Description = "A complete imported glTF 2.0 scene with geometry, hierarchy, materials, textures, cameras, and its original model representation")]
[AssetCapability<SceneSnapshotCapability>]
[AssetOutput<SceneSnapshotOutput>(
    "d25e6564-e47b-48c2-a1b8-700a42c9c7af",
    Symbol = "SceneSnapshot",
    Name = "Scene",
    PassDirections = AssetPassDirections.Input)]
public sealed partial class GltfModelAsset : ModelAsset;

[FeatherAssetType(
    "3d7b5b68-7c7e-4c07-b573-875976bf75ae",
    Name = "Layered Material",
    Description = "A material-domain example that references Texture without assuming a PBR runtime")]
public sealed partial class LayeredMaterialAsset : MaterialAsset
{
    [AssetInput(
        "e5ef0945-7f15-4c18-9fc8-8183538cfbbc",
        Name = "Base texture",
        Required = false,
        Role = AssetInputRole.Evaluation | AssetInputRole.Preview | AssetInputRole.Runtime,
        ChangeImpact = AssetChangeImpact.RuntimeCandidate)]
    public AssetRef<TextureAsset> BaseTexture { get; init; }

    [AssetInput(
        "042b544a-767b-4c9a-8beb-10cf1ee64212",
        Name = "Mix",
        Min = 0,
        Max = 1,
        Step = 0.01,
        Role = AssetInputRole.Evaluation | AssetInputRole.Preview | AssetInputRole.Runtime,
        ChangeImpact = AssetChangeImpact.RuntimeCandidate)]
    public float Mix { get; init; } = 1;
}

[FeatherAssetType(
    "bc28caf1-b94a-48fe-912c-c6e9d590e62f",
    Name = "Scene Document Projection",
    Description = "An immutable catalog projection of Scene-service authority")]
public sealed partial class SceneDocumentProjectionAsset : SceneAsset
{
    [AssetInput(
        "f8ea019b-0b98-4852-bbc7-2eca1e228030",
        Name = "Root actor template",
        Required = false,
        Role = AssetInputRole.Evaluation | AssetInputRole.Preview,
        ChangeImpact = AssetChangeImpact.ReevaluateOutputs)]
    public AssetRef<ActorAsset> RootActor { get; init; }
}

[FeatherAssetType(
    "380b11e2-22d5-4cac-8033-7b28d103d9c1",
    Name = "Actor Template Definition",
    Description = "A reusable Actor template that composes logical model and material references")]
public sealed partial class ActorTemplateDefinitionAsset : ActorAsset
{
    [AssetInput(
        "70023e80-e9cc-406c-9c82-c3324c9c58dc",
        Name = "3D model",
        Required = false,
        Role = AssetInputRole.Evaluation | AssetInputRole.Preview | AssetInputRole.Runtime,
        ChangeImpact = AssetChangeImpact.RuntimeCandidate)]
    public AssetRef<ModelAsset> Model { get; init; }

    [AssetInput(
        "7cbd1904-203a-4953-8974-fff7b49160f8",
        Name = "Material",
        Required = false,
        Role = AssetInputRole.Evaluation | AssetInputRole.Preview | AssetInputRole.Runtime,
        ChangeImpact = AssetChangeImpact.RuntimeCandidate)]
    public AssetRef<MaterialAsset> Material { get; init; }
}

// The following declarations demonstrate product-specific extensions without adding Atlas or SDF
// to Feather's core hierarchy. A first-party or package provider may realise the inherited Texture
// View slot for these types using the same provider boundary as any future custom Asset Type.
[FeatherAssetType(
    "c0d3edde-df30-450c-97eb-4f16eb2fb063",
    Name = "Texture Atlas",
    Description = "Official extension example for atlas metadata over a logical Texture source")]
public sealed partial class TextureAtlasAsset : TextureAsset
{
    [AssetInput(
        "7f1f185b-7170-4be8-a16b-766c3d418b97",
        Name = "Source texture",
        Role = AssetInputRole.Evaluation | AssetInputRole.Preview,
        ChangeImpact = AssetChangeImpact.ReevaluateOutputs)]
    public AssetRef<TextureAsset> Source { get; init; }

    [AssetInput(
        "ee88a50e-8d3e-4922-a5b8-006918d9ba81",
        Name = "Columns",
        Min = 1,
        Max = 4096,
        Step = 1)]
    public int Columns { get; init; } = 1;

    [AssetInput(
        "0de5afb3-782f-42ea-9de9-89b2c3dfe485",
        Name = "Rows",
        Min = 1,
        Max = 4096,
        Step = 1)]
    public int Rows { get; init; } = 1;
}

[FeatherAssetType(
    "886d0bfb-aa66-44f8-9a8b-13b849082281",
    Name = "Signed Distance Field Texture",
    Description = "Official extension example for an explicit SDF derivation contract")]
public sealed partial class SignedDistanceFieldTextureAsset : TextureAsset
{
    [AssetInput(
        "13f6f248-bf2a-4944-9afd-daa39b65f0a2",
        Name = "Source texture",
        Role = AssetInputRole.Evaluation | AssetInputRole.Preview,
        ChangeImpact = AssetChangeImpact.ReevaluateOutputs)]
    public AssetRef<TextureAsset> Source { get; init; }

    [AssetInput(
        "ef0a6268-8e40-4d02-ab77-d7f58d277ca7",
        Name = "Distance radius",
        Min = 1,
        Max = 256,
        Step = 1)]
    public float DistanceRadius { get; init; } = 16;
}
