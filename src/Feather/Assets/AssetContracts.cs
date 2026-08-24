using System.Diagnostics.CodeAnalysis;

namespace Feather.Assets;

/// <summary>
/// Nominal root for an author-defined Feather Asset Type. Asset identity and revision state live
/// in the owning catalog; an instance of this class is only the immutable authoring payload.
/// </summary>
public abstract class Asset
{
    protected Asset()
    {
    }
}

public readonly record struct AssetId
{
    public AssetId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

    public Guid Value { get; }

    public static AssetId Parse(string value) => new(AssetIdentity.ParseGuid(value, nameof(value)));

    public static bool TryParse([NotNullWhen(true)] string? value, out AssetId result)
    {
        if (AssetIdentity.TryParseGuid(value, out var parsed))
        {
            result = new AssetId(parsed);
            return true;
        }

        result = default;
        return false;
    }

    public override string ToString() => AssetIdentity.Format(Value, nameof(AssetId));
}

public readonly record struct AssetTypeId
{
    public AssetTypeId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

    public Guid Value { get; }

    public static AssetTypeId Parse(string value) => new(AssetIdentity.ParseGuid(value, nameof(value)));

    public static bool TryParse([NotNullWhen(true)] string? value, out AssetTypeId result)
    {
        if (AssetIdentity.TryParseGuid(value, out var parsed))
        {
            result = new AssetTypeId(parsed);
            return true;
        }

        result = default;
        return false;
    }

    public override string ToString() => AssetIdentity.Format(Value, nameof(AssetTypeId));
}

public readonly record struct AssetRevisionId
{
    public AssetRevisionId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

    public Guid Value { get; }

    public static AssetRevisionId Parse(string value) => new(AssetIdentity.ParseGuid(value, nameof(value)));

    public static bool TryParse([NotNullWhen(true)] string? value, out AssetRevisionId result)
    {
        if (AssetIdentity.TryParseGuid(value, out var parsed))
        {
            result = new AssetRevisionId(parsed);
            return true;
        }

        result = default;
        return false;
    }

    public override string ToString() => AssetIdentity.Format(Value, nameof(AssetRevisionId));
}

public readonly record struct AssetInputId
{
    public AssetInputId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

    public Guid Value { get; }

    public static AssetInputId Parse(string value) => new(AssetIdentity.ParseGuid(value, nameof(value)));

    public static bool TryParse([NotNullWhen(true)] string? value, out AssetInputId result)
    {
        if (AssetIdentity.TryParseGuid(value, out var parsed))
        {
            result = new AssetInputId(parsed);
            return true;
        }

        result = default;
        return false;
    }

    public override string ToString() => AssetIdentity.Format(Value, nameof(AssetInputId));
}

public readonly record struct AssetProductSlotId
{
    public AssetProductSlotId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

    public Guid Value { get; }

    public static AssetProductSlotId Parse(string value) => new(AssetIdentity.ParseGuid(value, nameof(value)));

    public static bool TryParse([NotNullWhen(true)] string? value, out AssetProductSlotId result)
    {
        if (AssetIdentity.TryParseGuid(value, out var parsed))
        {
            result = new AssetProductSlotId(parsed);
            return true;
        }

        result = default;
        return false;
    }

    public override string ToString() => AssetIdentity.Format(Value, nameof(AssetProductSlotId));
}

public readonly record struct AssetCapabilityId
{
    public AssetCapabilityId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

    public Guid Value { get; }

    public static AssetCapabilityId Parse(string value) => new(AssetIdentity.ParseGuid(value, nameof(value)));

    public static bool TryParse([NotNullWhen(true)] string? value, out AssetCapabilityId result)
    {
        if (AssetIdentity.TryParseGuid(value, out var parsed))
        {
            result = new AssetCapabilityId(parsed);
            return true;
        }

        result = default;
        return false;
    }

    public override string ToString() => AssetIdentity.Format(Value, nameof(AssetCapabilityId));
}

public readonly record struct AssetProviderId
{
    public AssetProviderId(Guid value) => Value = AssetIdentity.Require(value, nameof(value));

    public Guid Value { get; }

    public static AssetProviderId Parse(string value) => new(AssetIdentity.ParseGuid(value, nameof(value)));

    public static bool TryParse([NotNullWhen(true)] string? value, out AssetProviderId result)
    {
        if (AssetIdentity.TryParseGuid(value, out var parsed))
        {
            result = new AssetProviderId(parsed);
            return true;
        }

        result = default;
        return false;
    }

    public override string ToString() => AssetIdentity.Format(Value, nameof(AssetProviderId));
}

/// <summary>An immutable byte identity. Version 1 accepts SHA-256 only.</summary>
public readonly record struct AssetContentHash
{
    public AssetContentHash(string algorithm, string hex)
    {
        ArgumentNullException.ThrowIfNull(algorithm);
        ArgumentNullException.ThrowIfNull(hex);
        if (!string.Equals(algorithm, "sha256", StringComparison.Ordinal) ||
            hex.Length != 64 ||
            hex.Any(static character => character is not (>= '0' and <= '9') and not (>= 'a' and <= 'f')))
        {
            throw new ArgumentException(
                "Asset content hashes must be canonical lowercase sha256:<64 hex> values.",
                nameof(hex));
        }

        Algorithm = algorithm;
        Hex = hex;
    }

    public string Algorithm { get; }

    public string Hex { get; }

    public static AssetContentHash Parse(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (!TryParse(value, out var result))
        {
            throw new FormatException("Asset content hash must be canonical lowercase sha256:<64 hex>.");
        }

        return result;
    }

    public static bool TryParse([NotNullWhen(true)] string? value, out AssetContentHash result)
    {
        const string prefix = "sha256:";
        if (value is not null && value.StartsWith(prefix, StringComparison.Ordinal))
        {
            var hex = value[prefix.Length..];
            if (hex.Length == 64 &&
                hex.All(static character => character is (>= '0' and <= '9') or (>= 'a' and <= 'f')))
            {
                result = new AssetContentHash("sha256", hex);
                return true;
            }
        }

        result = default;
        return false;
    }

    public override string ToString()
        => Algorithm is not null && Hex is not null
            ? Algorithm + ":" + Hex
            : throw new InvalidOperationException("An uninitialized AssetContentHash has no identity.");
}

public readonly record struct AssetContractVersion
{
    public AssetContractVersion(ushort major, ushort minor = 0)
    {
        if (major == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(major), "Contract major version must be positive.");
        }

        Major = major;
        Minor = minor;
    }

    public ushort Major { get; }

    public ushort Minor { get; }

    public override string ToString() => $"{Major}.{Minor}";
}

public enum AssetRevisionPolicy
{
    FollowCompatibleCurrent = 0,
    PinnedExact = 1,
}

public readonly record struct AssetRevisionSelector
{
    private AssetRevisionSelector(
        AssetRevisionPolicy policy,
        AssetRevisionId? revisionId,
        AssetContentHash? revisionManifestHash)
    {
        Policy = policy;
        RevisionId = revisionId;
        RevisionManifestHash = revisionManifestHash;
    }

    public AssetRevisionPolicy Policy { get; }

    public AssetRevisionId? RevisionId { get; }

    public AssetContentHash? RevisionManifestHash { get; }

    public static AssetRevisionSelector FollowCompatibleCurrent { get; } =
        new(AssetRevisionPolicy.FollowCompatibleCurrent, null, null);

    public static AssetRevisionSelector PinnedExact(
        AssetRevisionId revisionId,
        AssetContentHash revisionManifestHash)
    {
        AssetIdentity.Require(revisionId.Value, nameof(revisionId));
        _ = revisionManifestHash.ToString();
        return new AssetRevisionSelector(
            AssetRevisionPolicy.PinnedExact,
            revisionId,
            revisionManifestHash);
    }
}

/// <summary>A durable logical selection. It never opens a file or exposes a runtime resource.</summary>
public readonly record struct AssetRef<TAsset>
    where TAsset : Asset
{
    private AssetRef(AssetId assetId, AssetRevisionSelector revision)
    {
        AssetIdentity.Require(assetId.Value, nameof(assetId));
        AssetId = assetId;
        Revision = revision;
    }

    public AssetId AssetId { get; }

    public AssetRevisionSelector Revision { get; }

    public static AssetRef<TAsset> Follow(AssetId assetId)
        => new(assetId, AssetRevisionSelector.FollowCompatibleCurrent);

    public static AssetRef<TAsset> Pin(
        AssetId assetId,
        AssetRevisionId revisionId,
        AssetContentHash revisionManifestHash)
        => new(assetId, AssetRevisionSelector.PinnedExact(revisionId, revisionManifestHash));
}

/// <summary>A durable logical output selection, not a prepared GPU/runtime handle.</summary>
public readonly record struct AssetOutputRef<TAsset, TOutput>
    where TAsset : Asset
    where TOutput : IAssetOutputContract
{
    public AssetOutputRef(AssetRef<TAsset> asset, AssetProductSlotId productSlotId)
    {
        AssetIdentity.Require(asset.AssetId.Value, nameof(asset));
        AssetIdentity.Require(productSlotId.Value, nameof(productSlotId));
        Asset = asset;
        ProductSlotId = productSlotId;
    }

    public AssetRef<TAsset> Asset { get; }

    public AssetProductSlotId ProductSlotId { get; }
}

public interface IAssetCapabilityContract;

public interface IAssetOutputContract;

[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class FeatherAssetTypeAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public string? Description { get; set; }

    public ushort ContractMajor { get; set; } = 1;

    public ushort ContractMinor { get; set; }

    public int PayloadSchemaVersion { get; set; } = 1;

    public bool Abstract { get; set; }
}

[Flags]
public enum AssetInputRole
{
    None = 0,
    Evaluation = 1 << 0,
    Runtime = 1 << 1,
    Preview = 1 << 2,
    Editor = 1 << 3,
    Provenance = 1 << 4,
}

public enum AssetChangeImpact
{
    MetadataOnly = 0,
    PreviewOnly = 1,
    ReevaluateOutputs = 2,
    RuntimeCandidate = 3,
}

[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
public sealed class AssetInputAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public string? Group { get; set; }

    public int Order { get; set; }

    public bool Required { get; set; } = true;

    public AssetInputRole Role { get; set; } = AssetInputRole.Evaluation;

    public AssetChangeImpact ChangeImpact { get; set; } = AssetChangeImpact.ReevaluateOutputs;

    public double Min { get; set; } = double.NaN;

    public double Max { get; set; } = double.NaN;

    public double Step { get; set; } = double.NaN;

    public int MinItems { get; set; } = -1;

    public int MaxItems { get; set; } = -1;

    public int MaxLength { get; set; } = -1;
}

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Interface, AllowMultiple = false, Inherited = false)]
public sealed class FeatherAssetCapabilityAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public ushort ContractMajor { get; set; } = 1;

    public ushort ContractMinor { get; set; }
}

[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class AssetCapabilityAttribute<TCapability> : Attribute
    where TCapability : IAssetCapabilityContract
{
    public ushort MinimumMajor { get; set; } = 1;

    public ushort MinimumMinor { get; set; }

    public bool Required { get; set; } = true;
}

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Interface, AllowMultiple = false, Inherited = false)]
public sealed class FeatherAssetOutputContractAttribute(string guid) : Attribute
{
    public string Guid { get; } = guid;

    public string? Name { get; set; }

    public ushort ContractMajor { get; set; } = 1;

    public ushort ContractMinor { get; set; }
}

[Flags]
public enum AssetPassDirections
{
    None = 0,
    Input = 1 << 0,
    Output = 1 << 1,
}

[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class AssetOutputAttribute<TOutput>(string guid) : Attribute
    where TOutput : IAssetOutputContract
{
    public string Guid { get; } = guid;

    public string Symbol { get; set; } = string.Empty;

    public string? Name { get; set; }

    public bool Required { get; set; } = true;

    public bool GraphOutput { get; set; } = true;

    public AssetPassDirections PassDirections { get; set; } = AssetPassDirections.Input;
}

public enum AssetProviderOperation
{
    Create = 0,
    Import = 1,
    Transform = 2,
    Build = 3,
    Preview = 4,
    RuntimeAdapter = 5,
}

public enum AssetProviderOwner
{
    AssetService = 0,
    IsolatedWorker = 1,
    RenderHost = 2,
}

public enum AssetProviderDeterminism
{
    Deterministic = 0,
    Seeded = 1,
    EnvironmentDependent = 2,
    NonDeterministic = 3,
}

[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class FeatherAssetProviderAttribute(string guid, AssetProviderOperation operation) : Attribute
{
    public string Guid { get; } = guid;

    public AssetProviderOperation Operation { get; } = operation;

    public string? Name { get; set; }

    public ushort ContractMajor { get; set; } = 1;

    public ushort ContractMinor { get; set; }

    public AssetProviderOwner Owner { get; set; } = AssetProviderOwner.AssetService;

    public AssetProviderDeterminism Determinism { get; set; } = AssetProviderDeterminism.Deterministic;
}

public interface IAssetCreator<TAsset>
    where TAsset : Asset
{
    ValueTask CreateAsync(AssetCreateContext<TAsset> context, CancellationToken cancellationToken);
}

public interface IAssetImporter<TAsset>
    where TAsset : Asset
{
    ValueTask ImportAsync(AssetImportContext<TAsset> context, CancellationToken cancellationToken);
}

public interface IAssetTransformer<TSource, TDestination>
    where TSource : Asset
    where TDestination : Asset
{
    ValueTask TransformAsync(
        AssetTransformContext<TSource, TDestination> context,
        CancellationToken cancellationToken);
}

public interface IAssetBuilder<TAsset>
    where TAsset : Asset
{
    ValueTask BuildAsync(AssetBuildContext<TAsset> context, CancellationToken cancellationToken);
}

public interface IAssetPreviewProvider<TAsset>
    where TAsset : Asset
{
    ValueTask PreparePreviewAsync(
        AssetPreviewContext<TAsset> context,
        CancellationToken cancellationToken);
}

public interface IAssetRuntimeAdapter<TOutput>
    where TOutput : IAssetOutputContract
{
    ValueTask PrepareAsync(
        AssetRuntimeAdapterContext<TOutput> context,
        CancellationToken cancellationToken);
}

/// <summary>
/// Host-created context base. It deliberately exposes no service locator, workspace path, catalog,
/// database, native handle, or Electron callback.
/// </summary>
public abstract class AssetProviderContext
{
    protected AssetProviderContext()
    {
    }

    public abstract AssetProviderId ProviderId { get; }

    public abstract AssetContentHash ProviderImplementationHash { get; }

    public abstract long MaximumInputBytes { get; }

    public abstract long MaximumOutputBytes { get; }
}

public abstract class AssetCreateContext<TAsset> : AssetProviderContext
    where TAsset : Asset
{
    public abstract TAsset Value { get; }
}

public abstract class AssetImportContext<TAsset> : AssetProviderContext
    where TAsset : Asset
{
    public abstract AssetSourceReadLease Source { get; }
}

public abstract class AssetTransformContext<TSource, TDestination> : AssetProviderContext
    where TSource : Asset
    where TDestination : Asset
{
    public abstract AssetRevisionSnapshot<TSource> Source { get; }
}

public abstract class AssetBuildContext<TAsset> : AssetProviderContext
    where TAsset : Asset
{
    public abstract AssetRevisionSnapshot<TAsset> Revision { get; }

    public abstract string Target { get; }
}

public abstract class AssetPreviewContext<TAsset> : AssetProviderContext
    where TAsset : Asset
{
    public abstract AssetRevisionSnapshot<TAsset> Revision { get; }

    public abstract string Profile { get; }

    public abstract string Target { get; }
}

public abstract class AssetRuntimeAdapterContext<TOutput> : AssetProviderContext
    where TOutput : IAssetOutputContract
{
    public abstract AssetProductSlotId ProductSlotId { get; }

    public abstract AssetContentHash ProductBuildManifestHash { get; }

    public abstract string Target { get; }
}

public abstract class AssetSourceReadLease : IAsyncDisposable
{
    protected AssetSourceReadLease()
    {
    }

    public abstract AssetContentHash ContentHash { get; }

    public abstract long Length { get; }

    public abstract string MediaType { get; }

    public abstract ValueTask<Stream> OpenReadAsync(CancellationToken cancellationToken);

    public abstract ValueTask DisposeAsync();
}

public sealed record AssetRevisionSnapshot<TAsset>(
    AssetId AssetId,
    AssetRevisionId RevisionId,
    AssetContentHash RevisionManifestHash,
    TAsset Value)
    where TAsset : Asset;

internal static class AssetIdentity
{
    public static Guid Require(Guid value, string parameterName)
        => value != Guid.Empty
            ? value
            : throw new ArgumentException("Asset identity must not be the empty GUID.", parameterName);

    public static Guid ParseGuid(string value, string parameterName)
    {
        ArgumentNullException.ThrowIfNull(value, parameterName);
        if (!TryParseGuid(value, out var result))
        {
            throw new FormatException("Asset identity must be a canonical lowercase UUID.");
        }

        return result;
    }

    public static bool TryParseGuid([NotNullWhen(true)] string? value, out Guid result)
    {
        if (value is not null &&
            Guid.TryParseExact(value, "D", out result) &&
            result != Guid.Empty &&
            string.Equals(value, result.ToString("D"), StringComparison.Ordinal))
        {
            return true;
        }

        result = default;
        return false;
    }

    public static string Format(Guid value, string typeName)
        => value != Guid.Empty
            ? value.ToString("D")
            : throw new InvalidOperationException($"An uninitialized {typeName} has no identity.");
}
