namespace Feather;

public interface IKernel1D
{
}

public interface IKernel2D
{
}

public interface IKernel3D
{
}

public interface IVertexShader<TVaryings>
    where TVaryings : unmanaged
{
}

public interface IFragmentShader<TVaryings>
    where TVaryings : unmanaged
{
}

public interface IFragmentShader<TVaryings, TOutput>
    where TVaryings : unmanaged
    where TOutput : unmanaged
{
}
