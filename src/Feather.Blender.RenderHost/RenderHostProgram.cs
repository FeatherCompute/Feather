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
                RenderHostResult result;
                do
                {
                    result = host.RenderOnce(options.RequestPath!);
                    WriteRenderResult(result);
                    if (result.NeedsMoreWork && result.TargetSamples == 0)
                    {
                        break;
                    }
                }
                while (result.NeedsMoreWork);
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

    /// <summary>
    /// How long after the last render the loop keeps checking for a new request at
    /// <see cref="ActivePollInterval"/> before falling back to the configured interval.
    /// </summary>
    private static readonly TimeSpan ActivePollWindow = TimeSpan.FromSeconds(1);

    /// <summary>
    /// The poll interval used while requests are still arriving.
    /// </summary>
    private static readonly TimeSpan ActivePollInterval = TimeSpan.FromMilliseconds(2);

    private static async Task<int> WatchAsync(
        RenderHostRunner host,
        RenderHostOptions options,
        CancellationToken cancellationToken)
    {
        RenderInputSignature? previous = null;
        RenderInputSignature? failed = null;
        RenderInputSignature? pending = null;
        RenderInputSignature? lastRead = null;
        var failureCount = 0;
        var retryAfter = DateTimeOffset.MinValue;
        var lastRender = DateTimeOffset.MinValue;
        WriteEvent("ready", new { requestPath = Path.GetFullPath(options.RequestPath!) });

        while (!cancellationToken.IsCancellationRequested)
        {
            var current = RenderInputSignature.TryRead(options.RequestPath!, lastRead);
            lastRead = current;
            var retryIsDue = current != failed || DateTimeOffset.UtcNow >= retryAfter;
            var continuesPendingRender = current is not null && current == pending;
            var rendered = false;
            if (current is not null && (current != previous || continuesPendingRender) && retryIsDue)
            {
                try
                {
                    var result = host.RenderOnce(options.RequestPath!);
                    previous = current;
                    pending = result.NeedsMoreWork ? current : null;
                    failed = null;
                    failureCount = 0;
                    rendered = true;
                    lastRender = DateTimeOffset.UtcNow;
                    WriteRenderResult(result);
                }
                catch (InvalidDataException exception)
                {
                    // Protocol errors are deterministic for this immutable request. A newly
                    // published request gets a new file signature and will be attempted normally.
                    previous = current;
                    pending = null;
                    failed = null;
                    failureCount = 0;
                    WriteError(exception);
                }
                catch (Exception exception) when (exception is not OperationCanceledException)
                {
                    failureCount = current == failed ? failureCount + 1 : 1;
                    failed = current;
                    pending = null;
                    var exponent = System.Math.Min(failureCount - 1, 5);
                    var retryMilliseconds = System.Math.Min(2000, 100 * (1 << exponent));
                    retryAfter = DateTimeOffset.UtcNow.AddMilliseconds(retryMilliseconds);
                    WriteError(exception);
                }
            }

            if (!rendered)
            {
                // Poll tightly for a short window after each render. While the viewport is being
                // dragged the next request lands at an arbitrary point inside the interval, so a
                // flat wait adds an average of half an interval -- and a full one whenever the
                // request arrives just after a check -- to a frame the host could already be
                // working on. That delay dominated the observed latency: a 21ms frame took 55ms to
                // reach the viewport at a 33ms interval. Once requests stop arriving the loop
                // relaxes back to the configured interval so an idle host stays cheap.
                var interval = DateTimeOffset.UtcNow - lastRender < ActivePollWindow
                    ? ActivePollInterval
                    : options.PollInterval;
                await Task.Delay(interval, cancellationToken);
            }
        }

        return 0;
    }

    private static void WriteEvent(string eventName, object value)
        => Console.WriteLine(JsonSerializer.Serialize(new { @event = eventName, value }, ProtocolJson.Options));

    private static void WriteRenderResult(RenderHostResult result)
        => WriteEvent(result.FramePublished ? "frame" : "progress", result);

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
        /// <param name="previous">
        /// The signature from the last poll, if any. When the request file itself is unchanged its
        /// resolved manifest path is too, so the parse can be skipped and only the manifest
        /// restatted. That keeps a tight poll loop from deserializing the request document several
        /// hundred times a second while nothing is happening.
        /// </param>
        public static RenderInputSignature? TryRead(
            string requestPath,
            RenderInputSignature? previous = null)
        {
            var request = FileSignature.TryRead(requestPath);
            if (request is null)
            {
                return null;
            }

            if (previous is not null && previous.Request == request)
            {
                var manifest = previous.ManifestPath is null
                    ? null
                    : FileSignature.TryRead(previous.ManifestPath);
                return manifest == previous.Manifest
                    ? previous
                    : new RenderInputSignature(request, previous.ManifestPath, manifest);
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
