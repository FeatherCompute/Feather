using System.Collections.Immutable;
using Feather.Math;

namespace Feather.Assets
{
    public readonly record struct AssetExtent3D
    {
        public AssetExtent3D(uint width, uint height, uint depth)
        {
            if (width == 0 || height == 0 || depth == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(width), "Asset extents must be positive.");
            }

            Width = width;
            Height = height;
            Depth = depth;
        }

        public uint Width { get; }

        public uint Height { get; }

        public uint Depth { get; }
    }

    public readonly record struct AssetBounds3D
    {
        public AssetBounds3D(float3 minimum, float3 maximum)
        {
            if (!Finite(minimum) || !Finite(maximum) ||
                minimum.X > maximum.X || minimum.Y > maximum.Y || minimum.Z > maximum.Z)
            {
                throw new ArgumentException("Asset bounds must be finite and ordered.", nameof(maximum));
            }

            Minimum = minimum;
            Maximum = maximum;
        }

        public float3 Minimum { get; }

        public float3 Maximum { get; }

        private static bool Finite(float3 value)
            => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
    }

    public readonly record struct AssetSemanticId
    {
        public AssetSemanticId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

        public Guid Value { get; }

        public override string ToString() => AssetIdentity.Format(Value, nameof(AssetSemanticId));
    }
}

namespace Feather.Assets.Graphics
{
    public readonly record struct TextureFormatId
    {
        public TextureFormatId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

        public Guid Value { get; }

        public override string ToString() => AssetIdentity.Format(Value, nameof(TextureFormatId));
    }

    public readonly record struct TextureColorSpaceId
    {
        public TextureColorSpaceId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

        public Guid Value { get; }

        public override string ToString() => AssetIdentity.Format(Value, nameof(TextureColorSpaceId));
    }

    public readonly record struct TextureSemanticId
    {
        public TextureSemanticId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

        public Guid Value { get; }

        public override string ToString() => AssetIdentity.Format(Value, nameof(TextureSemanticId));
    }

    public enum TextureDimension
    {
        D1 = 0,
        D2 = 1,
        D3 = 2,
        Cube = 3,
    }

    [Flags]
    public enum TextureViewFeatures
    {
        None = 0,
        Sampled = 1 << 0,
        Storage = 1 << 1,
        RenderTarget = 1 << 2,
        RandomPixelQuery = 1 << 3,
        Histogram = 1 << 4,
        SparseResidency = 1 << 5,
    }

    [FeatherAssetCapability(
        "7a10cc53-241d-4206-9631-e19d870ff370",
        Name = "Texture View")]
    public sealed class TextureViewCapability : IAssetCapabilityContract
    {
        public static AssetCapabilityId CapabilityId { get; } =
            AssetCapabilityId.Parse("7a10cc53-241d-4206-9631-e19d870ff370");
    }

    [FeatherAssetOutputContract(
        "59f5b9eb-20a0-47e1-a61f-aad9e0da2aee",
        Name = "Texture View")]
    public sealed record TextureViewOutput : IAssetOutputContract
    {
        public TextureViewOutput(
            int descriptorVersion,
            TextureDimension dimension,
            AssetExtent3D extent,
            TextureFormatId format,
            TextureColorSpaceId colorSpace,
            TextureSemanticId semantic,
            uint mipCount,
            uint layerCount,
            uint sampleCount,
            TextureViewFeatures features)
        {
            if (descriptorVersion <= 0 || !Enum.IsDefined(dimension) ||
                extent.Width == 0 || extent.Height == 0 || extent.Depth == 0 ||
                format.Value == Guid.Empty || colorSpace.Value == Guid.Empty || semantic.Value == Guid.Empty ||
                mipCount == 0 || layerCount == 0 || sampleCount == 0 ||
                !Enum.IsDefined(features & ~AllFeatures))
            {
                throw new ArgumentException("Texture View descriptor is invalid or incomplete.");
            }

            DescriptorVersion = descriptorVersion;
            Dimension = dimension;
            Extent = extent;
            Format = format;
            ColorSpace = colorSpace;
            Semantic = semantic;
            MipCount = mipCount;
            LayerCount = layerCount;
            SampleCount = sampleCount;
            Features = features;
        }

        public static Guid ContractId { get; } = Guid.Parse("59f5b9eb-20a0-47e1-a61f-aad9e0da2aee");

        public int DescriptorVersion { get; }
        public TextureDimension Dimension { get; }
        public AssetExtent3D Extent { get; }
        public TextureFormatId Format { get; }
        public TextureColorSpaceId ColorSpace { get; }
        public TextureSemanticId Semantic { get; }
        public uint MipCount { get; }
        public uint LayerCount { get; }
        public uint SampleCount { get; }
        public TextureViewFeatures Features { get; }

        private const TextureViewFeatures AllFeatures =
            TextureViewFeatures.Sampled |
            TextureViewFeatures.Storage |
            TextureViewFeatures.RenderTarget |
            TextureViewFeatures.RandomPixelQuery |
            TextureViewFeatures.Histogram |
            TextureViewFeatures.SparseResidency;
    }

    /// <summary>
    /// Canonical project Texture Type sampling surface. UV input and RGBA output use normalized
    /// semantic space; addressing, filtering, and color decoding remain explicit Asset inputs.
    /// </summary>
    public interface INormalizedTextureSampler
    {
        float4 Sample(float2 uv);
    }

    /// <summary>
    /// Nominal base for Assets that can realise the standard Texture View product on at least one
    /// declared target. It does not imply an image-file source, 2D shape, density, residency,
    /// color semantics, or a live GPU texture.
    /// </summary>
    [FeatherAssetType(
        "c2f0c619-d756-42f2-bb4b-a4ca48ab6dd2",
        Name = "Texture",
        Description = "A representation that can realise a compatible Texture View product on a declared target.",
        Abstract = true)]
    [AssetCapability<TextureViewCapability>]
    [AssetOutput<TextureViewOutput>(
        "d80529aa-1c69-47f2-ae46-63a7cb1e0916",
        Symbol = "TextureView",
        Name = "Texture View",
        PassDirections = AssetPassDirections.Input)]
    public abstract partial class TextureAsset : Asset;

    public readonly record struct MaterialDomainId
    {
        public MaterialDomainId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

        public Guid Value { get; }

        public override string ToString() => AssetIdentity.Format(Value, nameof(MaterialDomainId));
    }

    [FeatherAssetCapability(
        "6b2caa4f-cd3a-41c0-bf66-f9a4df88b821",
        Name = "Material Domain")]
    public sealed class MaterialDomainCapability : IAssetCapabilityContract
    {
        public static AssetCapabilityId CapabilityId { get; } =
            AssetCapabilityId.Parse("6b2caa4f-cd3a-41c0-bf66-f9a4df88b821");
    }

    [FeatherAssetOutputContract(
        "6ff31203-b0ce-46b1-a1ef-5f06d74e247a",
        Name = "Material Binding")]
    public sealed record MaterialBindingOutput : IAssetOutputContract
    {
        public MaterialBindingOutput(
            int descriptorVersion,
            MaterialDomainId domain,
            AssetContentHash bindingAbiHash,
            bool supportsStandardSurfacePreview)
        {
            if (descriptorVersion <= 0 || domain.Value == Guid.Empty)
                throw new ArgumentException("Material Binding descriptor is invalid.");
            _ = bindingAbiHash.ToString();
            DescriptorVersion = descriptorVersion;
            Domain = domain;
            BindingAbiHash = bindingAbiHash;
            SupportsStandardSurfacePreview = supportsStandardSurfacePreview;
        }

        public static Guid ContractId { get; } = Guid.Parse("6ff31203-b0ce-46b1-a1ef-5f06d74e247a");

        public int DescriptorVersion { get; }
        public MaterialDomainId Domain { get; }
        public AssetContentHash BindingAbiHash { get; }
        public bool SupportsStandardSurfacePreview { get; }
    }

    /// <summary>
    /// Nominal base for a value that participates in a declared material domain. It does not imply
    /// PBR, BSDF, RGB, rasterization, local illumination, or any visual preview.
    /// </summary>
    [FeatherAssetType(
        "293fd339-bf12-41dd-9f98-0519c9e17418",
        Name = "Material",
        Description = "A representation that participates in a declared material domain without imposing a shading model.",
        Abstract = true)]
    [AssetCapability<MaterialDomainCapability>]
    [AssetOutput<MaterialBindingOutput>(
        "0e6b623d-20be-4779-9fdf-7d4cfa662963",
        Symbol = "MaterialBinding",
        Name = "Material Binding",
        PassDirections = AssetPassDirections.Input)]
    public abstract partial class MaterialAsset : Asset;

    [FeatherAssetCapability(
        "b3bf0ffb-c538-4249-a30a-d7e2159f4778",
        Name = "3D Model Instantiation")]
    public sealed class ModelInstantiationCapability : IAssetCapabilityContract
    {
        public static AssetCapabilityId CapabilityId { get; } =
            AssetCapabilityId.Parse("b3bf0ffb-c538-4249-a30a-d7e2159f4778");
    }

    [FeatherAssetOutputContract(
        "c0395c08-d676-46e5-a3a4-d9f52f5ca4ec",
        Name = "3D Model Template")]
    public sealed record ModelTemplateOutput : IAssetOutputContract
    {
        public ModelTemplateOutput(
            int descriptorVersion,
            AssetBounds3D? bounds,
            bool hasHierarchy,
            bool hasSkins,
            bool hasAnimation,
            AssetContentHash templateAbiHash)
        {
            if (descriptorVersion <= 0) throw new ArgumentOutOfRangeException(nameof(descriptorVersion));
            _ = templateAbiHash.ToString();
            DescriptorVersion = descriptorVersion;
            Bounds = bounds;
            HasHierarchy = hasHierarchy;
            HasSkins = hasSkins;
            HasAnimation = hasAnimation;
            TemplateAbiHash = templateAbiHash;
        }

        public static Guid ContractId { get; } = Guid.Parse("c0395c08-d676-46e5-a3a4-d9f52f5ca4ec");

        public int DescriptorVersion { get; }
        public AssetBounds3D? Bounds { get; }
        public bool HasHierarchy { get; }
        public bool HasSkins { get; }
        public bool HasAnimation { get; }
        public AssetContentHash TemplateAbiHash { get; }
    }

    /// <summary>
    /// Nominal base for an instantiable three-dimensional model representation, including an
    /// optional hierarchy, geometry/material bindings, skins, and animation. It never means a
    /// learned or inference model; those belong to an explicitly named extension type.
    /// </summary>
    [FeatherAssetType(
        "8ade6d04-a60d-4a58-9ec6-33e039f3b6a0",
        Name = "3D Model",
        Description = "An instantiable three-dimensional model representation; never a learned or inference model.",
        Abstract = true)]
    [AssetCapability<ModelInstantiationCapability>]
    [AssetOutput<ModelTemplateOutput>(
        "23996109-e2a2-4f17-91f6-790cab1bd4dc",
        Symbol = "ModelTemplate",
        Name = "3D Model Template",
        PassDirections = AssetPassDirections.Input)]
    public abstract partial class ModelAsset : Asset;
}

namespace Feather.Assets.Scenes
{
    public readonly record struct SceneDocumentId
    {
        public SceneDocumentId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

        public Guid Value { get; }

        public override string ToString() => AssetIdentity.Format(Value, nameof(SceneDocumentId));
    }

    [FeatherAssetCapability(
        "5cd3ec1b-629a-46ef-9b8a-55f6e96102db",
        Name = "Scene Snapshot")]
    public sealed class SceneSnapshotCapability : IAssetCapabilityContract
    {
        public static AssetCapabilityId CapabilityId { get; } =
            AssetCapabilityId.Parse("5cd3ec1b-629a-46ef-9b8a-55f6e96102db");
    }

    [FeatherAssetOutputContract(
        "ecf5d49e-e75e-4fbf-be68-c0d07813f612",
        Name = "Scene Snapshot")]
    public sealed record SceneSnapshotOutput : IAssetOutputContract
    {
        public SceneSnapshotOutput(
            int descriptorVersion,
            SceneDocumentId sceneDocumentId,
            long sceneRevision,
            AssetContentHash sceneDocumentHash,
            AssetBounds3D? bounds,
            bool hasCameraProjection,
            bool hasHierarchyProjection,
            bool hasPicking,
            bool hasTimeline)
        {
            if (descriptorVersion <= 0 || sceneDocumentId.Value == Guid.Empty || sceneRevision <= 0)
                throw new ArgumentException("Scene Snapshot descriptor is invalid.");
            _ = sceneDocumentHash.ToString();
            DescriptorVersion = descriptorVersion;
            SceneDocumentId = sceneDocumentId;
            SceneRevision = sceneRevision;
            SceneDocumentHash = sceneDocumentHash;
            Bounds = bounds;
            HasCameraProjection = hasCameraProjection;
            HasHierarchyProjection = hasHierarchyProjection;
            HasPicking = hasPicking;
            HasTimeline = hasTimeline;
        }

        public static Guid ContractId { get; } = Guid.Parse("ecf5d49e-e75e-4fbf-be68-c0d07813f612");

        public int DescriptorVersion { get; }
        public SceneDocumentId SceneDocumentId { get; }
        public long SceneRevision { get; }
        public AssetContentHash SceneDocumentHash { get; }
        public AssetBounds3D? Bounds { get; }
        public bool HasCameraProjection { get; }
        public bool HasHierarchyProjection { get; }
        public bool HasPicking { get; }
        public bool HasTimeline { get; }
    }

    /// <summary>
    /// Catalog identity for a Scene-service document projection or an explicitly promoted immutable
    /// snapshot. It is never the mutable live hierarchy and does not move Scene command authority.
    /// </summary>
    [FeatherAssetType(
        "b934179f-2772-4419-afbc-a321888ec2ea",
        Name = "Scene",
        Description = "A catalog projection or immutable snapshot of a Scene-service document.",
        Abstract = true)]
    [AssetCapability<SceneSnapshotCapability>]
    [AssetOutput<SceneSnapshotOutput>(
        "d25e6564-e47b-48c2-a1b8-700a42c9c7af",
        Symbol = "SceneSnapshot",
        Name = "Scene Snapshot",
        PassDirections = AssetPassDirections.Input)]
    public abstract partial class SceneAsset : Asset;

    public readonly record struct ActorComponentContractId
    {
        public ActorComponentContractId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

        public Guid Value { get; }

        public override string ToString() => AssetIdentity.Format(Value, nameof(ActorComponentContractId));
    }

    [FeatherAssetCapability(
        "00e8076d-166c-4850-bf32-375aaed78602",
        Name = "Actor Instantiation")]
    public sealed class ActorInstantiationCapability : IAssetCapabilityContract
    {
        public static AssetCapabilityId CapabilityId { get; } =
            AssetCapabilityId.Parse("00e8076d-166c-4850-bf32-375aaed78602");
    }

    [FeatherAssetOutputContract(
        "81835893-81e0-458d-80fb-681c23b56d27",
        Name = "Actor Template")]
    public sealed record ActorTemplateOutput : IAssetOutputContract
    {
        public ActorTemplateOutput(
            int descriptorVersion,
            IEnumerable<ActorComponentContractId> componentContractIds,
            AssetBounds3D? bounds,
            bool hasPose,
            bool hasAnimation,
            AssetContentHash templateAbiHash)
        {
            ArgumentNullException.ThrowIfNull(componentContractIds);
            ImmutableArray<ActorComponentContractId> components = componentContractIds.ToImmutableArray();
            if (descriptorVersion <= 0 || components.Length > 4096 ||
                components.Any(static component => component.Value == Guid.Empty) ||
                components.Distinct().Count() != components.Length)
                throw new ArgumentException("Actor Template descriptor is invalid or unbounded.");
            _ = templateAbiHash.ToString();
            DescriptorVersion = descriptorVersion;
            ComponentContractIds = components;
            Bounds = bounds;
            HasPose = hasPose;
            HasAnimation = hasAnimation;
            TemplateAbiHash = templateAbiHash;
        }

        public static Guid ContractId { get; } = Guid.Parse("81835893-81e0-458d-80fb-681c23b56d27");

        public int DescriptorVersion { get; }
        public ImmutableArray<ActorComponentContractId> ComponentContractIds { get; }
        public AssetBounds3D? Bounds { get; }
        public bool HasPose { get; }
        public bool HasAnimation { get; }
        public AssetContentHash TemplateAbiHash { get; }
    }

    /// <summary>
    /// Reusable prototype/template that can be instantiated by Scene service. It is never a live
    /// Scene entity, and editing an instance never silently rewrites this Asset.
    /// </summary>
    [FeatherAssetType(
        "09dfd6df-e3b0-4bc2-882a-f42faf6be488",
        Name = "Actor",
        Description = "A reusable Scene-instantiation template, not a live entity.",
        Abstract = true)]
    [AssetCapability<ActorInstantiationCapability>]
    [AssetOutput<ActorTemplateOutput>(
        "4a209b60-c61b-441b-b0ba-87bf09671f86",
        Symbol = "ActorTemplate",
        Name = "Actor Template",
        PassDirections = AssetPassDirections.Input)]
    public abstract partial class ActorAsset : Asset;
}
