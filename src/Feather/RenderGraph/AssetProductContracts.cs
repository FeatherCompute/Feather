using Feather.Assets;

namespace Feather.RenderGraph;

/// <summary>
/// Generation-scoped prepared Asset product supplied by a render host. It is neither a durable
/// Asset reference nor a native handle and cannot be constructed or inspected by user code.
/// </summary>
public readonly record struct AssetOutputHandle<TOutput>
    where TOutput : IAssetOutputContract
{
    internal AssetOutputHandle(ulong generationToken)
    {
        GenerationToken = generationToken;
    }

    internal ulong GenerationToken { get; }
}

/// <summary>
/// Binds an Asset product socket to one stable Asset Type and product slot. The CLR type is only
/// an analyzer locator; generated persistence uses the type and slot GUIDs from their contracts.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
public sealed class AssetProductBindingAttribute(Type assetType, string productSlotGuid) : Attribute
{
    public Type AssetType { get; } = assetType;

    public string ProductSlotGuid { get; } = productSlotGuid;

    public bool Required { get; set; } = true;
}
