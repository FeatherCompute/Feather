namespace Feather.RenderGraph;

/// <summary>
/// Declares one project-owned Runtime Data Type. The annotated type is authoring metadata; its
/// logical resources are declared by <see cref="DataResourceAttribute"/> members and their typed
/// resource markers. Feather's generator publishes the immutable manifest consumed by Studio.
/// </summary>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, AllowMultiple = false, Inherited = false)]
public sealed class FeatherDataTypeAttribute(string dataTypeGuid) : Attribute
{
    public string DataTypeGuid { get; } = dataTypeGuid;

    public string? Name { get; set; }

    public ushort ContractMajor { get; set; } = 1;

    public ushort ContractMinor { get; set; }
}

[Flags]
public enum DataAccess
{
    Read = 1,
    Write = 2,
    ReadWrite = Read | Write,
}

public enum DataCreation
{
    BeforeGraph = 0,
    FirstUse = 1,
}

public enum DataUpdate
{
    Immutable = 0,
    OnDemand = 1,
    PassMutated = 2,
    PerFrame = 3,
}

public enum DataResourceLifetime
{
    View = 0,
    Graph = 1,
    Workspace = 2,
    Persistent = 3,
}

public enum DataFrames
{
    Single = 0,
    DoubleBuffered = 1,
}

/// <summary>
/// Describes one logical resource in a <see cref="FeatherDataTypeAttribute"/> declaration. The
/// member type selects Buffer/Texture/Probe/Cascade semantics while this attribute supplies stable
/// identity, lifecycle, shape, and admission bounds.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class DataResourceAttribute(string resourceGuid) : Attribute
{
    public string ResourceGuid { get; } = resourceGuid;

    public string? Name { get; set; }

    public DataAccess Access { get; set; } = DataAccess.ReadWrite;

    public DataCreation Creation { get; set; } = DataCreation.BeforeGraph;

    public DataUpdate Update { get; set; } = DataUpdate.PassMutated;

    public DataResourceLifetime Lifetime { get; set; } = DataResourceLifetime.Graph;

    public DataFrames Frames { get; set; } = DataFrames.Single;

    public long MaximumBytes { get; set; }

    public long ElementCount { get; set; }

    public int Width { get; set; }

    public int Height { get; set; }

    public int Depth { get; set; }

    public string? Format { get; set; }
}

/// <summary>Typed logical buffer declaration used only on a Data Type member.</summary>
public readonly record struct DataBuffer<T> where T : unmanaged;

/// <summary>Typed logical 1D texture declaration used only on a Data Type member.</summary>
public readonly record struct DataTexture1D<T> where T : unmanaged;

/// <summary>Typed logical 2D texture declaration used only on a Data Type member.</summary>
public readonly record struct DataTexture2D<T> where T : unmanaged;

/// <summary>Typed logical 3D texture declaration used only on a Data Type member.</summary>
public readonly record struct DataTexture3D<T> where T : unmanaged;

/// <summary>Typed logical probe-volume declaration used only on a Data Type member.</summary>
public readonly record struct DataProbeVolume<T> where T : unmanaged;

/// <summary>Typed logical radiance-cascade declaration used only on a Data Type member.</summary>
public readonly record struct DataRadianceCascade<T> where T : unmanaged;

/// <summary>Typed logical custom-resource declaration used only on a Data Type member.</summary>
public readonly record struct DataCustom<T> where T : unmanaged;

/// <summary>
/// Binds a pass socket to one exact Studio Data Type layout. The Data Instance remains one
/// logical graph object; RenderHost resolves its internal buffers/textures by stable resource ID.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false)]
public sealed class DataBindingAttribute(string dataTypeGuid, string layoutAbiHash) : Attribute
{
    public string DataTypeGuid { get; } = dataTypeGuid;

    public string LayoutAbiHash { get; } = layoutAbiHash;

    public ushort ContractMajor { get; set; } = 1;

    public ushort ContractMinor { get; set; }
}

/// <summary>
/// Identifies a Data Manager instance bound to a pass. It is intentionally opaque: passes ask
/// RenderContext for declared resources using stable Data Resource IDs instead of assuming a
/// fixed native struct or flattening the Data object into graph sockets.
/// </summary>
public readonly record struct DataHandle(ulong Value);

/// <summary>
/// Selects the logical frame version of a double-buffered Data resource. The Data Manager owns
/// physical allocations and swaps Current/Next only after the graph frame commits successfully.
/// </summary>
public enum DataFrameVersion
{
    Current = 0,
    Next = 1,
}
