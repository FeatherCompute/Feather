using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

const string MetadataKey = "Feather.PassManifest";

try
{
    var arguments = ParseArguments(args);
    var assemblyPath = Required(arguments, "assembly");
    var outputPath = Required(arguments, "output");
    var projectRoot = Required(arguments, "project-root");
    var feirDirectory = arguments.GetValueOrDefault("feir-directory") ?? "Generated/feather-ir";

    var assembly = Assembly.LoadFrom(Path.GetFullPath(assemblyPath));
    var encodedManifest = assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
        .SingleOrDefault(attribute => attribute.Key == MetadataKey)
        ?.Value;
    if (encodedManifest is null)
    {
        File.Delete(outputPath);
        return 0;
    }

    var manifestText = Encoding.UTF8.GetString(Convert.FromBase64String(encodedManifest));
    var manifest = JsonNode.Parse(manifestText)?.AsObject()
        ?? throw new InvalidDataException("Embedded Feather pass manifest is not a JSON object.");
    var relativeAssemblyPath = NormalizePath(Path.GetRelativePath(projectRoot, assemblyPath));
    feirDirectory = NormalizePath(feirDirectory).TrimEnd('/');

    manifest["assemblyPath"] = relativeAssemblyPath;
    manifest["feirDirectory"] = feirDirectory;
    foreach (var pass in manifest["passes"]?.AsArray().OfType<JsonObject>() ?? [])
    {
        var passGuid = pass["passGuid"]?.GetValue<string>()
            ?? throw new InvalidDataException("A Feather pass is missing passGuid.");
        pass["assemblyPath"] = relativeAssemblyPath;
        pass["feirPath"] = $"{feirDirectory}/{passGuid}.feir";
    }

    var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
    manifest["buildId"] = string.Empty;
    var hashInput = manifest.ToJsonString(jsonOptions);
    using var buildHasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    buildHasher.AppendData(Encoding.UTF8.GetBytes(hashInput));
    buildHasher.AppendData([0]);
    using (var assemblyStream = File.OpenRead(assemblyPath))
    {
        var buffer = new byte[81920];
        int bytesRead;
        while ((bytesRead = assemblyStream.Read(buffer, 0, buffer.Length)) > 0)
        {
            buildHasher.AppendData(buffer, 0, bytesRead);
        }
    }
    manifest["buildId"] = "sha256:" + Convert.ToHexString(
        buildHasher.GetHashAndReset()).ToLowerInvariant();
    var output = manifest.ToJsonString(jsonOptions) + Environment.NewLine;
    WriteIfChanged(Path.GetFullPath(outputPath), output);
    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine($"Feather manifest export failed: {exception.Message}");
    return 1;
}

static Dictionary<string, string> ParseArguments(string[] arguments)
{
    var result = new Dictionary<string, string>(StringComparer.Ordinal);
    for (var index = 0; index < arguments.Length; index += 2)
    {
        if (index + 1 >= arguments.Length || !arguments[index].StartsWith("--", StringComparison.Ordinal))
        {
            throw new ArgumentException("Arguments must be provided as --name value pairs.");
        }
        result[arguments[index][2..]] = arguments[index + 1];
    }
    return result;
}

static string Required(IReadOnlyDictionary<string, string> arguments, string name)
    => arguments.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value)
        ? value
        : throw new ArgumentException($"Missing required --{name} argument.");

static string NormalizePath(string value)
    => value.Replace('\\', '/');

static void WriteIfChanged(string path, string content)
{
    var directory = Path.GetDirectoryName(path)
        ?? throw new InvalidOperationException($"Manifest path has no parent directory: {path}");
    Directory.CreateDirectory(directory);
    if (File.Exists(path) && File.ReadAllText(path, Encoding.UTF8) == content)
    {
        return;
    }

    var temporaryPath = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
    try
    {
        File.WriteAllText(
            temporaryPath,
            content,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        File.Move(temporaryPath, path, overwrite: true);
    }
    finally
    {
        File.Delete(temporaryPath);
    }
}
