using System.Text;
using Feather.Native;

namespace Feather;

/// <summary>
/// Discovers Luisa compute devices and creates explicitly selected GPU contexts.
/// </summary>
public sealed class GpuRuntime
{
    private readonly Lazy<IReadOnlyList<GpuDeviceInfo>> devices;

    private GpuRuntime()
    {
        devices = new Lazy<IReadOnlyList<GpuDeviceInfo>>(DiscoverDevices);
    }

    /// <summary>
    /// Gets the process-default runtime.
    /// </summary>
    public static GpuRuntime Default { get; } = new();

    /// <summary>
    /// Creates an independent managed discovery object over the native Luisa runtime.
    /// </summary>
    public static GpuRuntime Create() => new();

    /// <summary>
    /// Enumerates devices exposed by the installed Luisa backends.
    /// </summary>
    public IReadOnlyList<GpuDeviceInfo> EnumerateDevices() => devices.Value;

    /// <summary>
    /// Gets the platform-default device (Metal on macOS, Vulkan elsewhere).
    /// </summary>
    public GpuDeviceInfo DefaultDevice => EnumerateDevices().FirstOrDefault(static device => device.IsDefault)
        ?? throw new InvalidOperationException("The platform-default Luisa device is unavailable.");

    /// <summary>
    /// Creates a context after validating the requested backend, device index, and required capabilities.
    /// </summary>
    public GpuContext CreateContext(GpuContextOptions options = default)
    {
        var backend = options.Backend == GpuBackend.Auto ? DefaultDevice.Backend : options.Backend;
        if (backend == GpuBackend.Auto)
        {
            throw new InvalidOperationException("The default Luisa backend could not be resolved.");
        }
        if (options.DeviceIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "DeviceIndex must be non-negative.");
        }

        var selected = EnumerateDevices().FirstOrDefault(device =>
            device.Backend == backend && device.DeviceIndex == options.DeviceIndex);
        if (selected is null)
        {
            throw new ArgumentException(
                $"Luisa device {options.DeviceIndex} is not available for backend '{GetBackendName(backend)}'.",
                nameof(options));
        }
        ValidateRequiredCapabilities(selected, options.RequiredCapabilities);

        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_create(
            selected.BackendName, checked((uint)selected.DeviceIndex), out var handle));
        try
        {
            return new GpuContext(handle, GetContextDevice(handle));
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    internal static GpuDeviceInfo GetContextDevice(FeContextHandle context)
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_device_info(context, out var info));
        return Convert(info);
    }

    private static IReadOnlyList<GpuDeviceInfo> DiscoverDevices()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_runtime_get_device_count(out var count));
        var result = new GpuDeviceInfo[checked((int)count)];
        for (uint ordinal = 0; ordinal < count; ordinal++)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_runtime_get_device_info(ordinal, out var info));
            result[checked((int)ordinal)] = Convert(info);
        }
        return Array.AsReadOnly(result);
    }

    private static void ValidateRequiredCapabilities(
        GpuDeviceInfo device,
        GpuRequiredCapabilities required)
    {
        ValidateCapability(required, GpuRequiredCapabilities.Bindless,
            device.Capabilities.BindlessCapacitySufficient, "bindless capacity", device);
        ValidateCapability(required, GpuRequiredCapabilities.Subgroup,
            device.Capabilities.Subgroup, "subgroup operations", device);
        ValidateCapability(required, GpuRequiredCapabilities.Quad,
            device.Capabilities.Quad, "quad operations", device);
    }

    private static void ValidateCapability(
        GpuRequiredCapabilities required,
        GpuRequiredCapabilities capability,
        GpuCapabilitySupport support,
        string displayName,
        GpuDeviceInfo device)
    {
        if ((required & capability) == 0 || support == GpuCapabilitySupport.Supported)
        {
            return;
        }
        throw new NotSupportedException(
            $"Device '{device.Name}' cannot satisfy required capability '{displayName}': {support}.");
    }

    private static unsafe GpuDeviceInfo Convert(FeDeviceInfo info)
    {
        string backendName;
        string deviceName;
        backendName = DecodeUtf8(info.BackendName, 16);
        deviceName = DecodeUtf8(info.DeviceName, 256);

        var parsedBackend = backendName switch
        {
            "vk" => GpuBackend.Vulkan,
            "metal" => GpuBackend.Metal,
            "cuda" => GpuBackend.Cuda,
            "hip" => GpuBackend.Hip,
            _ => throw new InvalidOperationException($"Native runtime returned unknown backend '{backendName}'.")
        };
        return new GpuDeviceInfo(
            parsedBackend,
            backendName,
            checked((int)info.DeviceIndex),
            deviceName,
            info.IsDefault != 0,
            new GpuDeviceCapabilities(
                info.ComputeWarpSize == 0 ? null : info.ComputeWarpSize,
                (GpuCapabilitySupport)info.BindlessCapacitySufficient,
                (GpuCapabilitySupport)info.Subgroup,
                (GpuCapabilitySupport)info.Quad));
    }

    private static unsafe string DecodeUtf8(byte* value, int capacity)
    {
        var length = 0;
        while (length < capacity && value[length] != 0)
        {
            length++;
        }
        return Encoding.UTF8.GetString(new ReadOnlySpan<byte>(value, length));
    }

    private static string GetBackendName(GpuBackend backend) => backend switch
    {
        GpuBackend.Vulkan => "vk",
        GpuBackend.Metal => "metal",
        GpuBackend.Cuda => "cuda",
        GpuBackend.Hip => "hip",
        _ => "auto"
    };
}

public enum GpuBackend
{
    Auto,
    Vulkan,
    Metal,
    Cuda,
    Hip
}

public enum GpuCapabilitySupport
{
    Unknown,
    Unsupported,
    Supported
}

[Flags]
public enum GpuRequiredCapabilities
{
    None = 0,
    Bindless = 1 << 0,
    Subgroup = 1 << 1,
    Quad = 1 << 2
}

public readonly record struct GpuContextOptions
{
    public GpuContextOptions()
    {
    }

    public GpuBackend Backend { get; init; } = GpuBackend.Auto;
    public int DeviceIndex { get; init; }
    public GpuRequiredCapabilities RequiredCapabilities { get; init; }
}

public sealed record GpuDeviceInfo(
    GpuBackend Backend,
    string BackendName,
    int DeviceIndex,
    string Name,
    bool IsDefault,
    GpuDeviceCapabilities Capabilities);

public sealed record GpuDeviceCapabilities(
    uint? ComputeWarpSize,
    GpuCapabilitySupport BindlessCapacitySufficient,
    GpuCapabilitySupport Subgroup,
    GpuCapabilitySupport Quad);
