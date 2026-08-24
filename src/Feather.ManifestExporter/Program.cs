using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

const string PassMetadataKey = "Feather.PassManifest";
const string AssetMetadataKey = "Feather.AssetManifest";

try
{
    var arguments = ParseArguments(args);
    var assemblyPath = Required(arguments, "assembly");
    var projectRoot = Required(arguments, "project-root");
    var passOutputPath = arguments.GetValueOrDefault("output");
    var assetOutputPath = arguments.GetValueOrDefault("asset-output");
    if (string.IsNullOrWhiteSpace(passOutputPath) && string.IsNullOrWhiteSpace(assetOutputPath))
    {
        throw new ArgumentException("At least one of --output or --asset-output is required.");
    }
    var feirDirectory = arguments.GetValueOrDefault("feir-directory") ?? "Generated/feather-ir";

    assemblyPath = Path.GetFullPath(assemblyPath);
    projectRoot = Path.GetFullPath(projectRoot);
    var assembly = Assembly.LoadFrom(assemblyPath);
    if (!string.IsNullOrWhiteSpace(passOutputPath))
    {
        ExportPassManifest(
            assembly,
            assemblyPath,
            Path.GetFullPath(passOutputPath),
            projectRoot,
            feirDirectory);
    }
    if (!string.IsNullOrWhiteSpace(assetOutputPath))
    {
        ExportAssetManifest(
            assembly,
            assemblyPath,
            Path.GetFullPath(assetOutputPath),
            projectRoot);
    }
    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine($"Feather manifest export failed: {exception.Message}");
    return 1;
}

static void ExportPassManifest(
    Assembly assembly,
    string assemblyPath,
    string outputPath,
    string projectRoot,
    string feirDirectory)
{
    var encodedManifest = Metadata(assembly, PassMetadataKey);
    if (encodedManifest is null)
    {
        DeleteIfPresent(outputPath);
        return;
    }

    var manifest = DecodeManifest(encodedManifest, "pass");
    var relativeAssemblyPath = NormalizePath(Path.GetRelativePath(projectRoot, assemblyPath));
    var outputDirectory = Path.GetDirectoryName(outputPath)
        ?? throw new InvalidOperationException($"Manifest path has no parent directory: {outputPath}");
    var relativeProjectRoot = NormalizePath(Path.GetRelativePath(outputDirectory, projectRoot));
    feirDirectory = NormalizePath(feirDirectory).TrimEnd('/');

    manifest["assemblyPath"] = relativeAssemblyPath;
    manifest["feirDirectory"] = feirDirectory;
    manifest["projectRoot"] = string.IsNullOrEmpty(relativeProjectRoot) ? "." : relativeProjectRoot;
    foreach (var pass in manifest["passes"]?.AsArray().OfType<JsonObject>() ?? [])
    {
        var passGuid = pass["passGuid"]?.GetValue<string>()
            ?? throw new InvalidDataException("A Feather pass is missing passGuid.");
        pass["assemblyPath"] = relativeAssemblyPath;
        var feirPath = $"{feirDirectory}/{passGuid}.feir";
        var resolvedFeirPath = Path.GetFullPath(
            Path.IsPathRooted(feirPath) ? feirPath : Path.Combine(projectRoot, feirPath));
        pass["feirPath"] = File.Exists(resolvedFeirPath) ? feirPath : string.Empty;
    }
    manifest["trainers"] = DiscoverTrainers(assembly);

    var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
    manifest["buildId"] = string.Empty;
    var hashInput = manifest.ToJsonString(jsonOptions);
    using var buildHasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    buildHasher.AppendData(Encoding.UTF8.GetBytes(hashInput));
    buildHasher.AppendData([0]);
    AppendFile(buildHasher, assemblyPath);
    manifest["buildId"] = Hash(buildHasher.GetHashAndReset());
    WriteIfChanged(outputPath, manifest.ToJsonString(jsonOptions) + Environment.NewLine);
}

static void ExportAssetManifest(
    Assembly assembly,
    string assemblyPath,
    string outputPath,
    string projectRoot)
{
    var encodedManifest = Metadata(assembly, AssetMetadataKey);
    if (encodedManifest is null)
    {
        DeleteIfPresent(outputPath);
        return;
    }

    var manifest = DecodeManifest(encodedManifest, "Asset");
    var generatorManifestHash = manifest["buildId"]?.GetValue<string>()
        ?? throw new InvalidDataException("Embedded Feather Asset manifest is missing buildId.");
    var assemblyHash = HashFile(assemblyPath);
    var exporterHash = HashFile(typeof(Program).Assembly.Location);
    manifest["generatorManifestHash"] = generatorManifestHash;
    manifest["assemblyPath"] = NormalizePath(Path.GetRelativePath(projectRoot, assemblyPath));
    manifest["assemblyHash"] = assemblyHash;
    manifest["toolchainHash"] = HashText(
        "Feather.AssetManifest.Exporter.v1\0" + generatorManifestHash + "\0" + exporterHash);

    foreach (var collectionName in new[]
             {
                 "assetTypes",
                 "capabilityContracts",
                 "outputContracts",
                 "providers",
             })
    {
        foreach (var item in manifest[collectionName]?.AsArray().OfType<JsonObject>() ?? [])
        {
            var source = item["source"]?.AsObject()
                ?? throw new InvalidDataException($"Asset manifest {collectionName} item is missing source.");
            var sourcePath = source["path"]?.GetValue<string>()
                ?? throw new InvalidDataException($"Asset manifest {collectionName} source is missing path.");
            var normalizedPath = NormalizeSourcePath(projectRoot, sourcePath);
            source["path"] = normalizedPath;
            source["documentHash"] = HashFile(Path.Combine(projectRoot, normalizedPath));
        }
    }

    var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
    manifest["buildId"] = string.Empty;
    manifest["manifestHash"] = string.Empty;
    using (var buildHasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
    {
        buildHasher.AppendData(Encoding.UTF8.GetBytes(manifest.ToJsonString(jsonOptions)));
        buildHasher.AppendData([0]);
        AppendFile(buildHasher, assemblyPath);
        manifest["buildId"] = Hash(buildHasher.GetHashAndReset());
    }
    manifest["manifestHash"] = HashText(manifest.ToJsonString(jsonOptions));
    WriteIfChanged(outputPath, manifest.ToJsonString(jsonOptions) + Environment.NewLine);
}

static string? Metadata(Assembly assembly, string key)
    => assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
        .SingleOrDefault(attribute => attribute.Key == key)
        ?.Value;

static JsonObject DecodeManifest(string encodedManifest, string kind)
{
    var manifestText = Encoding.UTF8.GetString(Convert.FromBase64String(encodedManifest));
    return JsonNode.Parse(manifestText)?.AsObject()
        ?? throw new InvalidDataException($"Embedded Feather {kind} manifest is not a JSON object.");
}

static string NormalizeSourcePath(string projectRoot, string sourcePath)
{
    if (string.IsNullOrWhiteSpace(sourcePath) || Path.IsPathRooted(sourcePath))
    {
        throw new InvalidDataException("Asset source paths must be non-empty and project-relative.");
    }

    var normalizedRoot = Path.GetFullPath(projectRoot).TrimEnd(Path.DirectorySeparatorChar) +
        Path.DirectorySeparatorChar;
    var fullPath = Path.GetFullPath(Path.Combine(projectRoot, sourcePath));
    if (!fullPath.StartsWith(normalizedRoot, StringComparison.Ordinal))
    {
        throw new InvalidDataException("Asset source path escapes the project root.");
    }
    if (!File.Exists(fullPath))
    {
        throw new FileNotFoundException("Asset source file was not found.", fullPath);
    }
    return NormalizePath(Path.GetRelativePath(projectRoot, fullPath));
}

static string HashFile(string path)
{
    using var hasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    AppendFile(hasher, path);
    return Hash(hasher.GetHashAndReset());
}

static string HashText(string value)
    => Hash(SHA256.HashData(Encoding.UTF8.GetBytes(value)));

static string Hash(byte[] value)
    => "sha256:" + Convert.ToHexString(value).ToLowerInvariant();

static void AppendFile(IncrementalHash hasher, string path)
{
    using var stream = File.OpenRead(path);
    var buffer = new byte[81920];
    int bytesRead;
    while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
    {
        hasher.AppendData(buffer, 0, bytesRead);
    }
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

static void DeleteIfPresent(string path)
{
    if (File.Exists(path))
    {
        File.Delete(path);
    }
}

static JsonArray DiscoverTrainers(Assembly assembly)
{
    const string trainerAttributeName = "Feather.NN.FeatherTrainerAttribute";
    const string trainingJobInterfaceName = "Feather.NN.ITrainingJob";
    const string parameterAttributeName = "Feather.RenderGraph.ParameterAttribute";
    var trainers = new JsonArray();
    foreach (var type in assembly.GetTypes().OrderBy(type => type.FullName, StringComparer.Ordinal))
    {
        var attribute = type.CustomAttributes.SingleOrDefault(item =>
            item.AttributeType.FullName == trainerAttributeName);
        if (attribute is null)
        {
            continue;
        }

        var guid = attribute.ConstructorArguments.FirstOrDefault().Value as string ?? string.Empty;
        var name = NamedAttributeValue(attribute, "Name") as string ?? type.Name;
        var category = NamedAttributeValue(attribute, "Category") as string ?? "Training";
        var version = NamedAttributeValue(attribute, "Version") is int declaredVersion ? declaredVersion : 1;
        var parameters = new JsonArray();
        foreach (var member in type.GetMembers(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                     .Where(member => member is FieldInfo or PropertyInfo)
                     .OrderBy(member => member.MetadataToken))
        {
            var parameter = member.CustomAttributes.SingleOrDefault(item =>
                item.AttributeType.FullName == parameterAttributeName);
            if (parameter is null)
            {
                continue;
            }

            var memberType = member is FieldInfo field ? field.FieldType : ((PropertyInfo)member).PropertyType;
            var defaultValue = NamedAttributeValue(parameter, "DefaultValue");
            var minimum = NamedAttributeValue(parameter, "Min");
            var maximum = NamedAttributeValue(parameter, "Max");
            var item = new JsonObject
            {
                ["parameterGuid"] = parameter.ConstructorArguments.FirstOrDefault().Value as string ?? string.Empty,
                ["name"] = NamedAttributeValue(parameter, "Name") as string ?? member.Name,
                ["type"] = ManifestTypeName(memberType),
                ["defaultValue"] = AttributeJsonValue(defaultValue),
            };
            if (minimum is double min && double.IsFinite(min)) item["min"] = min;
            if (maximum is double max && double.IsFinite(max)) item["max"] = max;
            parameters.Add(item);
        }

        trainers.Add(new JsonObject
        {
            ["trainerGuid"] = guid,
            ["typeName"] = type.FullName ?? type.Name,
            ["displayName"] = name,
            ["category"] = category,
            ["version"] = version,
            ["implementsTrainingJob"] = type.GetInterfaces().Any(item => item.FullName == trainingJobInterfaceName),
            ["parameters"] = parameters,
        });
    }
    return trainers;
}

static object? NamedAttributeValue(CustomAttributeData attribute, string name)
{
    foreach (var argument in attribute.NamedArguments)
    {
        if (argument.MemberName == name)
        {
            return argument.TypedValue.Value;
        }
    }
    return default;
}

static string ManifestTypeName(Type type)
    => Type.GetTypeCode(type) switch
    {
        TypeCode.Boolean => "bool",
        TypeCode.Int32 => "int",
        TypeCode.Single => "float",
        TypeCode.Double => "double",
        TypeCode.String => "string",
        _ => type.FullName ?? type.Name,
    };

static JsonNode? AttributeJsonValue(object? value)
    => value switch
    {
        null => null,
        bool item => JsonValue.Create(item),
        byte item => JsonValue.Create(item),
        short item => JsonValue.Create(item),
        int item => JsonValue.Create(item),
        long item => JsonValue.Create(item),
        float item => JsonValue.Create(item),
        double item => JsonValue.Create(item),
        string item => JsonValue.Create(item),
        _ => JsonValue.Create(value.ToString()),
    };
