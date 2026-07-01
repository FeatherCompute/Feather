using System.Runtime.InteropServices;
using System.Text;

namespace Feather.Native;

public static class NativeMethods
{
    private const string LibraryName = "Feather.NativeRuntime";
    private const string ContractExportName = "fe_ir_bridge_contract_version";
    private static int resolverInitialized;
    private static int runtimeShutdown;
    private static int processExiting;

    static NativeMethods()
    {
        EnsureResolverInitialized();
        AppDomain.CurrentDomain.ProcessExit += static (_, _) => ProcessExitShutdownRuntime();
    }

    public static void EnsureResolverInitialized()
    {
        if (Interlocked.Exchange(ref resolverInitialized, 1) == 1)
        {
            return;
        }

        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, ResolveNativeLibrary);
    }

    public static bool Succeeded(this FeResult result) => result == FeResult.Ok;

    public static bool IsProcessExiting => Volatile.Read(ref processExiting) != 0;

    public static void ShutdownRuntime()
    {
        if (Interlocked.Exchange(ref runtimeShutdown, 1) == 1)
        {
            return;
        }

        try
        {
            _ = fe_runtime_shutdown();
        }
        catch (DllNotFoundException)
        {
        }
        catch (EntryPointNotFoundException)
        {
        }
    }

    private static void ProcessExitShutdownRuntime()
    {
        Volatile.Write(ref processExiting, 1);
        _ = Interlocked.Exchange(ref runtimeShutdown, 1);
    }

    public static void ThrowIfFailed(FeResult result)
    {
        if (result == FeResult.Ok)
        {
            return;
        }

        throw new FeatherNativeException(result, GetLastError());
    }

    public static string GetLastError()
    {
        var required = UIntPtr.Zero;
        var result = fe_get_last_error(IntPtr.Zero, UIntPtr.Zero, out required);
        if (result != FeResult.Ok && required == UIntPtr.Zero)
        {
            return "Native Feather call failed and no native error string was available.";
        }

        var size = checked((int)required) + 1;
        var buffer = new byte[size];
        unsafe
        {
            fixed (byte* ptr = buffer)
            {
                _ = fe_get_last_error((IntPtr)ptr, (UIntPtr)buffer.Length, out _);
            }
        }

        var length = Array.IndexOf(buffer, (byte)0);
        if (length < 0)
        {
            length = buffer.Length;
        }

        return Encoding.UTF8.GetString(buffer, 0, length);
    }

    private static IntPtr ResolveNativeLibrary(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, LibraryName, StringComparison.Ordinal))
        {
            return IntPtr.Zero;
        }

        foreach (var candidate in EnumerateNativeLibraryCandidates(assembly))
        {
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out var handle))
            {
                if (NativeLibrary.TryGetExport(handle, ContractExportName, out _))
                {
                    return handle;
                }

                NativeLibrary.Free(handle);
            }
        }

        return IntPtr.Zero;
    }

    private static IEnumerable<string> EnumerateNativeLibraryCandidates(System.Reflection.Assembly assembly)
    {
        var overridePath = Environment.GetEnvironmentVariable("FEATHER_NATIVE_LIBRARY");
        if (!string.IsNullOrWhiteSpace(overridePath))
        {
            yield return overridePath;
        }

        var fileName = GetNativeLibraryFileName();
        var baseDirectory = AppContext.BaseDirectory;
        yield return Path.Combine(baseDirectory, fileName);
        yield return Path.Combine(baseDirectory, "native", fileName);

        var assemblyDirectory = Path.GetDirectoryName(assembly.Location);
        if (!string.IsNullOrWhiteSpace(assemblyDirectory))
        {
            yield return Path.Combine(assemblyDirectory, fileName);
        }

        foreach (var root in EnumerateAncestorDirectories(baseDirectory))
        {
            yield return Path.Combine(root, "native", "build", fileName);
            yield return Path.Combine(root, "artifacts", "native-assets", "runtimes", RuntimeInformation.RuntimeIdentifier, "native", fileName);
            yield return Path.Combine(root, "runtimes", RuntimeInformation.RuntimeIdentifier, "native", fileName);
        }
    }

    private static IEnumerable<string> EnumerateAncestorDirectories(string start)
    {
        var directory = new DirectoryInfo(start);
        while (directory is not null)
        {
            yield return directory.FullName;
            directory = directory.Parent;
        }
    }

    private static string GetNativeLibraryFileName()
    {
        if (OperatingSystem.IsWindows())
        {
            return "feather_native.dll";
        }

        if (OperatingSystem.IsMacOS())
        {
            return "libfeather.dylib";
        }

        return "libfeather.so";
    }

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_context_get_default(out FeContextHandle out_context);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_context_initialize(FeContextHandle context);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_context_shutdown(FeContextHandle context);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "fe_context_shutdown")]
    public static extern FeResult fe_context_shutdown_raw(IntPtr context);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_context_get_backend_type(FeContextHandle context, out uint out_backend);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_context_get_caps(FeContextHandle context, out FeBackendCaps out_caps);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_get_last_error(IntPtr buffer, UIntPtr buffer_size, out UIntPtr out_required_size);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_runtime_shutdown();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_create(in FeWindowDesc desc, out FeWindowHandle out_window);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_destroy(IntPtr window);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_is_open(FeWindowHandle window, [MarshalAs(UnmanagedType.I1)] out bool out_is_open);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_close(FeWindowHandle window);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_poll_events(FeWindowHandle window);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_wait_events(FeWindowHandle window);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_poll_event(FeWindowHandle window, out FeWindowEvent out_event, [MarshalAs(UnmanagedType.I1)] out bool out_has_event);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_get_size(FeWindowHandle window, out uint out_width, out uint out_height);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_set_title(FeWindowHandle window, [MarshalAs(UnmanagedType.LPUTF8Str)] string title);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_set_vsync(FeWindowHandle window, [MarshalAs(UnmanagedType.I1)] bool enabled);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_is_key_down(FeWindowHandle window, uint key, [MarshalAs(UnmanagedType.I1)] out bool out_is_down);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_is_mouse_down(FeWindowHandle window, uint mouse_button, [MarshalAs(UnmanagedType.I1)] out bool out_is_down);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_get_mouse_position(FeWindowHandle window, out int out_x, out int out_y);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_get_mouse_scroll(FeWindowHandle window, out float out_x, out float out_y);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_window_present_pixels(FeWindowHandle window, IntPtr pixels, uint width, uint height);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture_presenter_create(FeWindowHandle window, out FeTexturePresenterHandle out_presenter);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture_presenter_destroy(IntPtr presenter);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture_presenter_present_texture(FeTexturePresenterHandle presenter, FeTextureHandle texture, uint mode);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture_presenter_present_pixels(FeTexturePresenterHandle presenter, IntPtr pixels, uint width, uint height);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_buffer_create(FeContextHandle context, in FeBufferDesc desc, IntPtr initial_data, out FeBufferHandle out_buffer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_buffer_destroy(IntPtr buffer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_buffer_upload(FeBufferHandle buffer, ulong offset, ulong size, IntPtr data);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_buffer_download(FeBufferHandle buffer, ulong offset, ulong size, IntPtr out_data);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_buffer_map(FeBufferHandle buffer, uint mode, out IntPtr out_ptr);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_buffer_unmap(FeBufferHandle buffer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture2d_create(FeContextHandle context, in FeTexture2DDesc desc, IntPtr initial_data, out FeTextureHandle out_texture);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture3d_create(FeContextHandle context, in FeTexture3DDesc desc, IntPtr initial_data, out FeTextureHandle out_texture);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture_destroy(IntPtr texture);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture2d_upload(FeTextureHandle texture, uint x, uint y, uint width, uint height, IntPtr data);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture2d_download(FeTextureHandle texture, uint x, uint y, uint width, uint height, IntPtr out_data);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture3d_upload(FeTextureHandle texture, uint x, uint y, uint z, uint width, uint height, uint depth, IntPtr data);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture3d_download(FeTextureHandle texture, uint x, uint y, uint z, uint width, uint height, uint depth, IntPtr out_data);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_texture_generate_mipmaps(FeTextureHandle texture);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_sampler_create(FeContextHandle context, in FeSamplerDesc desc, out FeSamplerHandle out_sampler);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_sampler_destroy(IntPtr sampler);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_create_from_ir(FeContextHandle context, in FeKernelCreateDesc desc, out FeKernelHandle out_kernel);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_destroy(IntPtr kernel);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_bind_buffer(FeKernelHandle kernel, uint binding, FeBufferHandle buffer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_bind_texture(FeKernelHandle kernel, uint binding, FeTextureHandle texture);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_bind_sampler(FeKernelHandle kernel, uint binding, FeSamplerHandle sampler);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_set_push_constants(FeKernelHandle kernel, IntPtr data, ulong size);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_dispatch(
        FeKernelHandle kernel,
        uint group_x,
        uint group_y,
        uint group_z,
        uint logical_x,
        uint logical_y,
        uint logical_z,
        [MarshalAs(UnmanagedType.I1)] bool wait);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_get_glsl(FeKernelHandle kernel, IntPtr buffer, UIntPtr buffer_size, out UIntPtr out_required_size);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_get_optimized_glsl(FeKernelHandle kernel, IntPtr buffer, UIntPtr buffer_size, out UIntPtr out_required_size);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_get_last_dispatch_path(FeKernelHandle kernel, out FeDispatchPath out_path);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_get_ad_gradient_count(FeKernelHandle kernel, out uint out_count);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_get_ad_gradient_info(FeKernelHandle kernel, uint index, out FeADGradientInfo out_info);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_read_ad_gradient(FeKernelHandle kernel, uint index, ulong offset, ulong size, IntPtr out_data);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_reduce_ad_gradient_to_buffer(FeKernelHandle kernel, uint index, FeBufferHandle destination, ulong destinationOffset, ulong destinationSize);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_kernel_get_ad_backward_glsl(FeKernelHandle kernel, IntPtr buffer, UIntPtr buffer_size, out UIntPtr out_required_size);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_create_from_ir(FeContextHandle context, in FeGraphicsPipelineCreateDesc desc, out FeGraphicsPipelineHandle out_pipeline);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_destroy(IntPtr pipeline);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_set_vertex_buffer(FeGraphicsPipelineHandle pipeline, FeBufferHandle buffer, uint stride);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_set_index_buffer(FeGraphicsPipelineHandle pipeline, FeBufferHandle buffer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_bind_buffer(FeGraphicsPipelineHandle pipeline, uint binding, FeBufferHandle buffer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_bind_texture(FeGraphicsPipelineHandle pipeline, uint binding, FeTextureHandle texture);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_bind_sampler(FeGraphicsPipelineHandle pipeline, uint binding, FeSamplerHandle sampler);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_set_push_constants(FeGraphicsPipelineHandle pipeline, IntPtr data, ulong size);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_draw(FeGraphicsPipelineHandle pipeline, FeTextureHandle targetHandle, FeTextureHandle depth_target, uint vertex_count, [MarshalAs(UnmanagedType.I1)] bool wait);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_draw_ex(FeGraphicsPipelineHandle pipeline, in FeGraphicsDrawDesc desc);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_draw_indexed(
        FeGraphicsPipelineHandle pipeline,
        FeTextureHandle color_target,
        FeTextureHandle depth_target,
        FeBufferHandle index_buffer,
        uint index_count,
        [MarshalAs(UnmanagedType.I1)] bool wait);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_graphics_pipeline_get_last_dispatch_path(FeGraphicsPipelineHandle pipeline, out FeDispatchPath out_path);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_profiler_set_enabled([MarshalAs(UnmanagedType.I1)] bool enabled);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_profiler_is_enabled([MarshalAs(UnmanagedType.I1)] out bool out_enabled);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_profiler_clear();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_profiler_get_total_time(out double out_total_time_ms);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_profiler_query([MarshalAs(UnmanagedType.LPUTF8Str)] string name, out FeProfilerQueryResult out_result);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_profiler_get_formatted(IntPtr buffer, UIntPtr buffer_size, out UIntPtr out_required_size);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern uint fe_ir_bridge_contract_version();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern FeResult fe_ir_validate(IntPtr ir_data, ulong ir_size);
}

public sealed class FeatherNativeException(FeResult result, string message) : Exception(message)
{
    public FeResult Result { get; } = result;
}
