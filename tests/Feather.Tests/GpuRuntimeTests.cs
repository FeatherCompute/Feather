namespace Feather.Tests;

public class GpuRuntimeTests
{
    [Fact]
    public void DeviceDiscoveryReportsRealLuisaDevicesAndPlatformDefault()
    {
        var runtime = GpuRuntime.Create();

        var devices = runtime.EnumerateDevices();

        Assert.NotEmpty(devices);
        Assert.All(devices, static device =>
        {
            Assert.Contains(device.BackendName, new[] { "vk", "metal", "cuda", "hip" });
            Assert.True(device.DeviceIndex >= 0);
            Assert.False(string.IsNullOrWhiteSpace(device.Name));
            Assert.Equal(GpuCapabilitySupport.Unknown, device.Capabilities.BindlessCapacitySufficient);
            Assert.Equal(GpuCapabilitySupport.Unknown, device.Capabilities.Subgroup);
            Assert.Equal(GpuCapabilitySupport.Unknown, device.Capabilities.Quad);
        });
        Assert.Equal(devices.Count, devices.Select(static device =>
            (device.Backend, device.DeviceIndex)).Distinct().Count());

        var expectedBackend = OperatingSystem.IsMacOS() ? GpuBackend.Metal : GpuBackend.Vulkan;
        Assert.Equal(expectedBackend, runtime.DefaultDevice.Backend);
        Assert.Equal(0, runtime.DefaultDevice.DeviceIndex);
        Assert.Equal(runtime.DefaultDevice, GPU.Context.Device);
    }

    [Fact]
    public void ContextSelectionRejectsUnavailableDeviceAndUnknownRequirement()
    {
        var runtime = GpuRuntime.Create();
        var defaultDevice = runtime.DefaultDevice;

        var invalid = new GpuContextOptions
        {
            Backend = defaultDevice.Backend,
            DeviceIndex = int.MaxValue
        };
        var unavailable = Assert.Throws<ArgumentException>(() => runtime.CreateContext(invalid));
        Assert.Contains("not available", unavailable.Message, StringComparison.Ordinal);

        var unknownRequirement = new GpuContextOptions
        {
            Backend = defaultDevice.Backend,
            DeviceIndex = defaultDevice.DeviceIndex,
            RequiredCapabilities = GpuRequiredCapabilities.Quad
        };
        var unsupported = Assert.Throws<NotSupportedException>(() => runtime.CreateContext(unknownRequirement));
        Assert.Contains("Unknown", unsupported.Message, StringComparison.Ordinal);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void DefaultLuisaDeviceCanCreateExplicitContext()
    {
        var runtime = GpuRuntime.Create();
        var expected = runtime.DefaultDevice;

        using var context = runtime.CreateContext();

        Assert.Equal(expected.Backend, context.Device.Backend);
        Assert.Equal(expected.BackendName, context.Device.BackendName);
        Assert.Equal(expected.DeviceIndex, context.Device.DeviceIndex);
        Assert.Equal(expected.Name, context.Device.Name);
        Assert.True(context.Device.Capabilities.ComputeWarpSize > 0);
    }
}
