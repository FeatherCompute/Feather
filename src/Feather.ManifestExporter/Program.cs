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

    assemblyPath = Path.GetFullPath(assemblyPath);
    outputPath = Path.GetFullPath(outputPath);
    projectRoot = Path.GetFullPath(projectRoot);
    var assembly = Assembly.LoadFrom(assemblyPath);
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
    WriteIfChanged(outputPath, output);
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
