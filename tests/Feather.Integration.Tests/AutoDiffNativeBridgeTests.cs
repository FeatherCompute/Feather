using System.Text;
using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

namespace Feather.Integration.Tests;

public class AutoDiffNativeBridgeTests
{
    [Fact]
    public unsafe void ADBackwardGeneratesMergedGlsl()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<ScalarQuadraticAdKernel>(GPU.Context);
        var kernel = new ScalarQuadraticAdKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        DispatchRetained(gpuKernel, kernel, 1);

        var glsl = NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_kernel_get_ad_backward_glsl(gpuKernel.Handle, buffer, length, out required));

        Assert.Contains("Backward pass", glsl, StringComparison.Ordinal);
        Assert.Contains("_ad_grad_fe_0_data", glsl, StringComparison.Ordinal);
        Assert.Contains("layout(std430", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ADKernelSupportsOneShotGeneratedDispatch()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);

        GpuKernel.Dispatch(
            GPU.Context,
            new ScalarQuadraticAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()),
            new GpuDispatchSize(1, 1, 1),
            wait: true);

        Assert.Equal([9f], loss.ToArray());
    }

    [Fact]
    public void ADBackwardKeepsFusedMultiplyAddPeepholeDisabled()
    {
        using var scale = GPU.CreateBuffer<float>([2f]);
        using var bias = GPU.CreateBuffer<float>([1f]);
        using var input = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new ScalarMultiplyAddAdKernel(
            scale.AsReadWrite(), bias.AsReadWrite(), input.AsReadOnly(), loss.AsReadWrite()));

        ad.Backward(1);

        Assert.DoesNotContain("fma(", ad.GetBackwardGLSL(), StringComparison.Ordinal);
        Assert.InRange(ad.Gradients.Get<float>("scale")[0], 41.999f, 42.001f);
        Assert.InRange(ad.Gradients.Get<float>("bias")[0], 13.999f, 14.001f);
        Assert.Equal([49f], loss.ToArray());
    }

    [Fact]
    public void ADBackwardMergedGlslGuardsPaddedWorkgroupLanes()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(64);
        using var gpuKernel = GpuKernel.Create<SharedScalarPaddedLaneAdKernel>(GPU.Context);
        var kernel = new SharedScalarPaddedLaneAdKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        DispatchRetained(gpuKernel, kernel, 64);

        var glsl = NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_kernel_get_ad_backward_glsl(gpuKernel.Handle, buffer, length, out required));

        Assert.Equal(DispatchPath.TypedEasyGpu, gpuKernel.LastDispatchPath);
        Assert.Contains("if (gl_GlobalInvocationID.x >= 64u) return;", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ADBackwardReducesOnlyLogicalLanesWhenWorkgroupIsPadded()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(64);
        using var adKernel = GPU.CreateADKernel(new SharedScalarPaddedLaneAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));
        using var reducedGradient = GPU.CreateBuffer<float>(1);

        adKernel.Backward(64);
        adKernel.CopyGradientToBuffer("parameters", reducedGradient);

        Assert.Equal(DispatchPath.TypedEasyGpu, adKernel.LastDispatchPath);
        Assert.InRange(reducedGradient.ToArray()[0], 383.99f, 384.01f);
    }

    [Fact]
    public unsafe void ADBackwardWritesReadableGradientAndPreservesParameterBuffer()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<ScalarQuadraticAdKernel>(GPU.Context);
        var kernel = new ScalarQuadraticAdKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        DispatchRetained(gpuKernel, kernel, 1);

        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_ad_gradient_count(gpuKernel.Handle, out var count));
        Assert.Equal(1u, count);

        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_ad_gradient_info(gpuKernel.Handle, 0, out var info));
        Assert.Equal("parameters", FixedString(info.Name, 128));
        Assert.Equal("parameters", FixedString(info.ResourceName, 128));
        Assert.Equal("float", FixedString(info.ElementType, 64));
        Assert.Equal("fe_0", FixedString(info.EasyGpuName, 64));
        Assert.Equal(0u, info.SourceBinding);
        Assert.Equal(1u, info.ElementCount);
        Assert.Equal(4u, info.ElementStride);
        Assert.Equal(1u, info.ComponentCount);
        Assert.True(info.ByteSize >= sizeof(float));

        var gradients = new float[checked((int)(info.ByteSize / sizeof(float)))];
        fixed (float* ptr = gradients)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_read_ad_gradient(
                gpuKernel.Handle,
                0,
                0,
                info.ByteSize,
                (IntPtr)ptr));
        }

        Assert.InRange(gradients[0], 5.99f, 6.01f);
        Assert.Equal([3f], parameters.ToArray());
    }

    [Fact]
    public void ADBackwardMergedGlslIncludesIfControlFlow()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<IfElseAdInspectionKernel>(GPU.Context);
        var kernel = new IfElseAdInspectionKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        DispatchRetained(gpuKernel, kernel, 1);

        var glsl = NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_kernel_get_ad_backward_glsl(gpuKernel.Handle, buffer, length, out required));

        Assert.Equal(DispatchPath.TypedEasyGpu, gpuKernel.LastDispatchPath);
        Assert.Contains("Backward pass", glsl, StringComparison.Ordinal);
        Assert.Contains("if (", glsl, StringComparison.Ordinal);
        Assert.Contains("_ad_grad_fe_0_data", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ADBackwardMergedGlslIncludesReverseForControlFlow()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<ForLoopAdInspectionKernel>(GPU.Context);
        var kernel = new ForLoopAdInspectionKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        DispatchRetained(gpuKernel, kernel, 1);

        var glsl = NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_kernel_get_ad_backward_glsl(gpuKernel.Handle, buffer, length, out required));

        Assert.Equal(DispatchPath.TypedEasyGpu, gpuKernel.LastDispatchPath);
        Assert.Contains("Backward pass", glsl, StringComparison.Ordinal);
        Assert.Contains("for (", glsl, StringComparison.Ordinal);
        Assert.Contains("_ad_grad_fe_0_data", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ADBackwardMergesRepeatedLoopNamesAndCompoundMaxOperands()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<RepeatedLoopNameAndMaxAdKernel>(GPU.Context);
        var kernel = new RepeatedLoopNameAndMaxAdKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        DispatchRetained(gpuKernel, kernel, 1);

        var glsl = NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_kernel_get_ad_backward_glsl(gpuKernel.Handle, buffer, length, out required));

        Assert.Equal(DispatchPath.TypedEasyGpu, gpuKernel.LastDispatchPath);
        Assert.Contains("Backward pass", glsl, StringComparison.Ordinal);
        Assert.Contains("step(", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("= ;", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ADDispatchFailsClearlyWithoutLoss()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var gpuKernel = GpuKernel.Create<MissingLossAdKernel>(GPU.Context);
        var kernel = new MissingLossAdKernel(parameters.AsReadWrite());

        var ex = Assert.Throws<FeatherNativeException>(() => DispatchRetained(gpuKernel, kernel, 1));
        Assert.Equal(FeResult.ErrorUnsupported, ex.Result);
        Assert.Contains("loss", ex.Message, StringComparison.OrdinalIgnoreCase);

        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_ad_gradient_count(gpuKernel.Handle, out var count));
        Assert.Equal(0u, count);
    }

    [Fact]
    public void ADDispatchFailsClearlyForWhileLoop()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<WhileLoopAdKernel>(GPU.Context);
        var kernel = new WhileLoopAdKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        var ex = Assert.Throws<FeatherNativeException>(() => DispatchRetained(gpuKernel, kernel, 1));

        Assert.Equal(FeResult.ErrorUnsupported, ex.Result);
        Assert.Contains("while loops are not supported", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ADDispatchFailsClearlyForDoWhileLoop()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<DoWhileLoopAdKernel>(GPU.Context);
        var kernel = new DoWhileLoopAdKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        var ex = Assert.Throws<FeatherNativeException>(() => DispatchRetained(gpuKernel, kernel, 1));

        Assert.Equal(FeResult.ErrorUnsupported, ex.Result);
        Assert.Contains("do-while loops are not supported", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ADDispatchFailsClearlyForBreak()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<BreakAdKernel>(GPU.Context);
        var kernel = new BreakAdKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        var ex = Assert.Throws<FeatherNativeException>(() => DispatchRetained(gpuKernel, kernel, 1));

        Assert.Equal(FeResult.ErrorUnsupported, ex.Result);
        Assert.Contains("break statements are not supported", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ADDispatchFailsClearlyForContinue()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var gpuKernel = GpuKernel.Create<ContinueAdKernel>(GPU.Context);
        var kernel = new ContinueAdKernel(parameters.AsReadWrite(), loss.AsReadWrite());

        var ex = Assert.Throws<FeatherNativeException>(() => DispatchRetained(gpuKernel, kernel, 1));

        Assert.Equal(FeResult.ErrorUnsupported, ex.Result);
        Assert.Contains("continue statements are not supported", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static void DispatchRetained<TKernel>(GpuKernel gpuKernel, TKernel kernel, int count)
        where TKernel : struct, IGeneratedKernel<TKernel>
        => GpuKernel.Dispatch(GPU.Context, gpuKernel, kernel, new GpuDispatchSize(count, 1, 1), wait: true);

    private static unsafe string FixedString(byte* value, int length)
    {
        var span = new ReadOnlySpan<byte>(value, length);
        var nul = span.IndexOf((byte)0);
        if (nul >= 0)
        {
            span = span[..nul];
        }

        return Encoding.UTF8.GetString(span);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ScalarQuadraticAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float l = p * p;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ScalarMultiplyAddAdKernel(
    ReadWriteBuffer<float> scale,
    ReadWriteBuffer<float> bias,
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float y = (scale[0] * input[0]) + bias[0];
        float l = y * y;
        loss[0] = l;
        ADMarker.Parameter(scale[0]);
        ADMarker.Parameter(bias[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[AutoDiff]
public readonly partial struct SharedScalarPaddedLaneAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[0];
        float l = p * p;
        loss[i] = l;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct MissingLossAdKernel(ReadWriteBuffer<float> parameters) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        parameters[i] = p * 2f;
        ADMarker.Parameter(parameters[i]);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct IfElseAdInspectionKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float y;
        if (p > 0f)
        {
            y = p * 2f;
        }
        else
        {
            y = p * 3f;
        }

        float l = y * y;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ForLoopAdInspectionKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float sum = 0f;
        for (int j = 0; j < 5; j = j + 1)
        {
            sum = sum + p;
        }

        float l = sum * sum;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct RepeatedLoopNameAndMaxAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float maxValue = -1000f;
        float sum = 0f;
        for (int v = 0; v < 3; v = v + 1)
        {
            float term = p * (v + 1);
            maxValue = ShaderMath.Max(maxValue, term * 0.5f);
            sum = sum + term;
        }

        for (int v = 0; v < 2; v = v + 1)
        {
            float term = p + v;
            maxValue = ShaderMath.Max(maxValue, term * 0.25f);
            sum = sum + term;
        }

        float l = (maxValue + sum) * (maxValue + sum);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct WhileLoopAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float sum = 0f;
        int j = 0;
        while (j < 2)
        {
            sum = sum + p;
            j = j + 1;
        }

        float l = sum * sum;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct DoWhileLoopAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float sum = 0f;
        int j = 0;
        do
        {
            sum = sum + p;
            j = j + 1;
        }
        while (j < 2);

        float l = sum * sum;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct BreakAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float sum = 0f;
        for (int j = 0; j < 5; j = j + 1)
        {
            if (j > 1)
            {
                break;
            }

            sum = sum + p;
        }

        float l = sum * sum;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ContinueAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float sum = 0f;
        for (int j = 0; j < 5; j = j + 1)
        {
            if (j == 1)
            {
                continue;
            }

            sum = sum + p;
        }

        float l = sum * sum;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}
