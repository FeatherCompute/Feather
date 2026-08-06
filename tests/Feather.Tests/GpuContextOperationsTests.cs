namespace Feather.Tests;

public class GpuContextOperationsTests
{
    [Fact]
    public void ExplicitContextFacadeRequiresAContext()
    {
        Assert.Throws<ArgumentNullException>(() => GPU.WithContext(null!));
    }
}
