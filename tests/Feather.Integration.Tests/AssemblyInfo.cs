using Xunit;

// Native profiler state is process-wide, so integration tests run serially to avoid cross-test timing pollution.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
