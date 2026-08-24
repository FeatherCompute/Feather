using Feather.Assets;

namespace AssetContracts;

[FeatherAssetCapability(
    "04052f23-dc9c-4f0a-9770-ebd35bfddbbb",
    Name = "Field Sampling",
    ContractMajor = 1)]
public sealed class FieldSampling : IAssetCapabilityContract;

[FeatherAssetOutputContract(
    "81b8755c-a712-4c21-9e76-0e13a48eda43",
    Name = "Dense Scalar Field",
    ContractMajor = 1)]
public sealed class DenseScalarField : IAssetOutputContract;

[FeatherAssetType(
    "2d0a7b51-4bf5-4ec2-89ea-2a184a073d0f",
    Name = "Spatial Field",
    Abstract = true)]
public abstract partial class SpatialFieldAsset : Asset;

[FeatherAssetType(
    "878827ac-7fe1-4990-acad-554923b696c8",
    Name = "Gradient Field",
    Description = "A nontraditional field representation used to qualify open Asset contracts",
    PayloadSchemaVersion = 1)]
[AssetCapability<FieldSampling>]
[AssetOutput<DenseScalarField>(
    "32087aaa-22f8-4033-95f3-f86a4654614b",
    Symbol = "DenseField",
    Name = "Dense Field",
    PassDirections = AssetPassDirections.Input | AssetPassDirections.Output)]
public sealed partial class GradientFieldAsset : SpatialFieldAsset
{
    [AssetInput(
        "0228c70f-7456-416f-807d-f4cd4b96e859",
        Name = "Scale",
        Min = 0,
        Max = 16,
        Step = 0.01,
        Role = AssetInputRole.Evaluation | AssetInputRole.Runtime,
        ChangeImpact = AssetChangeImpact.RuntimeCandidate)]
    public float Scale { get; init; } = 1;

    [AssetInput(
        "79574624-0838-4cef-8962-aab26ad1ea26",
        Name = "Label",
        MaxLength = 64,
        Role = AssetInputRole.Editor,
        ChangeImpact = AssetChangeImpact.MetadataOnly)]
    public string Label { get; init; } = "Gradient Field";
}

[FeatherAssetType(
    "97c5237c-2ee8-4858-8881-f5e4726116da",
    Name = "Opaque Archive",
    Description = "A valid Asset Type that deliberately declares no preview or runtime output")]
public sealed partial class OpaqueArchiveAsset : Asset
{
    [AssetInput(
        "a27a4d2c-6431-4847-8051-88101bb84a1c",
        Name = "Label",
        MaxLength = 64,
        Role = AssetInputRole.Editor,
        ChangeImpact = AssetChangeImpact.MetadataOnly)]
    public string Label { get; init; } = "Archive";
}
