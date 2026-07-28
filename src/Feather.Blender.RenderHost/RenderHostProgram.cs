using System.Text.Json;

namespace Feather.Blender.RenderHost;

internal static class RenderHostProgram
{
    public static async Task<int> RunAsync(string[] args)
    {
        RenderHostOptions options;
        try
        {
            options = RenderHostOptions.Parse(args);
        }
        catch (ArgumentException exception)
        {
            Console.Error.WriteLine(exception.Message);
            Console.Error.WriteLine(RenderHostOptions.Usage);
            return 2;
        }

        if (options.ShowHelp)
        {
            Console.WriteLine(RenderHostOptions.Usage);
            return 0;
        }

        using var cancellation = new CancellationTokenSource();
        ConsoleCancelEventHandler cancelHandler = (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            cancellation.Cancel();
        };
        Console.CancelKeyPress += cancelHandler;

        try
        {
            using var host = new RenderHostRunner();
            if (!options.Watch)
            {
                var result = host.RenderOnce(options.RequestPath!);
                WriteEvent("frame", result);
                return 0;
            }

            return await WatchAsync(host, options, cancellation.Token);
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested)
        {
            return 0;
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            WriteError(exception);
            return 1;
        }
        finally
        {
            Console.CancelKeyPress -= cancelHandler;
        }
    }

    private static async Task<int> WatchAsync(
        RenderHostRunner host,
        RenderHostOptions options,
        CancellationToken cancellationToken)
    {
        RenderInputSignature? previous = null;
        RenderInputSignature? failed = null;
        var failureCount = 0;
        var retryAfter = DateTimeOffset.MinValue;
        WriteEvent("ready", new { requestPath = Path.GetFullPath(options.RequestPath!) });

        while (!cancellationToken.IsCancellationRequested)
        {
            var current = RenderInputSignature.TryRead(options.RequestPath!);
            var retryIsDue = current != failed || DateTimeOffset.UtcNow >= retryAfter;
            if (current is not null && current != previous && retryIsDue)
            {
                try
                {
                    var result = host.RenderOnce(options.RequestPath!);
                    previous = current;
                    failed = null;
                    failureCount = 0;
                    WriteEvent("frame", result);
                }
                catch (InvalidDataException exception)
                {
                    // Protocol errors are deterministic for this immutable request. A newly
                    // published request gets a new file signature and will be attempted normally.
                    previous = current;
                    failed = null;
                    failureCount = 0;
                    WriteError(exception);
                }
                catch (Exception exception) when (exception is not OperationCanceledException)
                {
                    failureCount = current == failed ? failureCount + 1 : 1;
                    failed = current;
                    var exponent = System.Math.Min(failureCount - 1, 5);
                    var retryMilliseconds = System.Math.Min(2000, 100 * (1 << exponent));
                    retryAfter = DateTimeOffset.UtcNow.AddMilliseconds(retryMilliseconds);
                    WriteError(exception);
                }
            }

            await Task.Delay(options.PollInterval, cancellationToken);
        }

        return 0;
    }

    private static void WriteEvent(string eventName, object value)
        => Console.WriteLine(JsonSerializer.Serialize(new { @event = eventName, value }, ProtocolJson.Options));

    private static void WriteError(Exception exception)
        => Console.Error.WriteLine(JsonSerializer.Serialize(
            new { @event = "error", error = exception.GetType().Name, message = exception.Message },
            ProtocolJson.Options));

    private sealed record FileSignature(long Length, long LastWriteTicks)
    {
        public static FileSignature? TryRead(string path)
        {
            try
            {
                var info = new FileInfo(path);
                return info.Exists ? new FileSignature(info.Length, info.LastWriteTimeUtc.Ticks) : null;
            }
            catch (IOException)
            {
                return null;
            }
        }
    }

    private sealed record RenderInputSignature(
        FileSignature Request,
        string? ManifestPath,
        FileSignature? Manifest)
    {
        public static RenderInputSignature? TryRead(string requestPath)
        {
            var request = FileSignature.TryRead(requestPath);
            if (request is null)
            {
                return null;
            }

            try
            {
                var resolved = RenderRequest.Load(requestPath);
                return new RenderInputSignature(
                    request,
                    resolved.ManifestPath,
                    resolved.ManifestPath is null ? null : FileSignature.TryRead(resolved.ManifestPath));
            }
            catch (Exception exception) when (exception is
                IOException or
                UnauthorizedAccessException or
                InvalidDataException or
                ArgumentException or
                OverflowException)
            {
                return new RenderInputSignature(request, null, null);
            }
        }
    }
}

internal sealed record RenderHostOptions(
    string? RequestPath,
    bool Watch,
    TimeSpan PollInterval,
    bool ShowHelp)
{
    public const string Usage = "Usage: Feather.Blender.RenderHost --request <render-request.json> [--watch] [--poll-ms <10-5000>]";

    public static RenderHostOptions Parse(string[] args)
    {
        string? requestPath = null;
        var watch = false;
        var pollMilliseconds = 33;
        var showHelp = false;

        for (var index = 0; index < args.Length; index++)
        {
            switch (args[index])
            {
                case "--request":
                    requestPath = RequireValue(args, ref index, "--request");
                    break;
                case "--watch":
                    watch = true;
                    break;
                case "--poll-ms":
                    var value = RequireValue(args, ref index, "--poll-ms");
                    if (!int.TryParse(value, out pollMilliseconds) || pollMilliseconds is < 10 or > 5000)
                    {
                        throw new ArgumentException("--poll-ms must be an integer between 10 and 5000.");
                    }
                    break;
                case "--help" or "-h":
                    showHelp = true;
                    break;
                default:
                    throw new ArgumentException($"Unknown argument: {args[index]}");
            }
        }

        if (!showHelp && string.IsNullOrWhiteSpace(requestPath))
        {
            throw new ArgumentException("--request is required.");
        }

        return new RenderHostOptions(requestPath, watch, TimeSpan.FromMilliseconds(pollMilliseconds), showHelp);
    }

    private static string RequireValue(string[] args, ref int index, string option)
    {
        if (++index >= args.Length || string.IsNullOrWhiteSpace(args[index]))
        {
            throw new ArgumentException($"{option} requires a value.");
        }

        return args[index];
    }
}
