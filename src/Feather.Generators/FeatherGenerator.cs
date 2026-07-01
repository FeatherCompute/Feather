using System.Text;
using Feather.Generators.IR;
using Feather.Generators.Model;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Feather.Generators;

[Generator(LanguageNames.CSharp)]
public sealed class FeatherGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var computeModels = context.SyntaxProvider.ForAttributeWithMetadataName(
                "Feather.KernelAttribute",
                static (node, _) => node is StructDeclarationSyntax,
                static (ctx, ct) => ShaderModelFactory.Create(ctx, ct))
            .Where(static model => model is not null)
            .Select(static (model, _) => model!);

        var vertexModels = context.SyntaxProvider.ForAttributeWithMetadataName(
                "Feather.VertexShaderAttribute",
                static (node, _) => node is StructDeclarationSyntax,
                static (ctx, ct) => ShaderModelFactory.Create(ctx, ct))
            .Where(static model => model is not null)
            .Select(static (model, _) => model!);

        var fragmentModels = context.SyntaxProvider.ForAttributeWithMetadataName(
                "Feather.FragmentShaderAttribute",
                static (node, _) => node is StructDeclarationSyntax,
                static (ctx, ct) => ShaderModelFactory.Create(ctx, ct))
            .Where(static model => model is not null)
            .Select(static (model, _) => model!);

        var gpuStructModels = context.SyntaxProvider.ForAttributeWithMetadataName(
                "Feather.GpuStructAttribute",
                static (node, _) => node is TypeDeclarationSyntax,
                static (ctx, ct) => GpuStructModelFactory.Create(ctx, ct))
            .Where(static model => model is not null)
            .Select(static (model, _) => model!);

        context.RegisterSourceOutput(gpuStructModels, static (productionContext, model) =>
        {
            var hasErrors = false;
            foreach (var diagnostic in GpuStructModelFactory.Validate(model))
            {
                hasErrors |= diagnostic.Severity == DiagnosticSeverity.Error;
                productionContext.ReportDiagnostic(diagnostic);
            }

            if (hasErrors)
            {
                return;
            }

            productionContext.AddSource($"{model.Name}.Feather.GpuStruct.g.cs", SourceText.From(GenerateGpuStruct(model), Encoding.UTF8));
        });

        context.RegisterSourceOutput(computeModels, static (productionContext, model) =>
        {
            var hasDiagnostics = false;
            foreach (var diagnostic in ShaderModelFactory.Validate(model))
            {
                hasDiagnostics = true;
                productionContext.ReportDiagnostic(diagnostic);
            }

            if (model.Kind is ShaderKind.Compute1D or ShaderKind.Compute2D or ShaderKind.Compute3D)
            {
                if (!hasDiagnostics && model.TypedIrSection is { Length: > 0 })
                {
                    productionContext.AddSource($"{model.Name}.Feather.g.cs", SourceText.From(GenerateKernel(model), Encoding.UTF8));
                }
            }
        });

        var graphicsPipelines = vertexModels.Collect().Combine(fragmentModels.Collect());
        context.RegisterSourceOutput(graphicsPipelines, static (productionContext, pair) =>
        {
            var invalidShaders = new HashSet<ShaderModel>();
            foreach (var model in pair.Left.Concat(pair.Right))
            {
                foreach (var diagnostic in ShaderModelFactory.Validate(model))
                {
                    if (diagnostic.Severity == DiagnosticSeverity.Error)
                    {
                        invalidShaders.Add(model);
                    }

                    productionContext.ReportDiagnostic(diagnostic);
                }
            }

            var validVertices = pair.Left.Where(model => !invalidShaders.Contains(model)).ToArray();
            var validFragments = pair.Right.Where(model => !invalidShaders.Contains(model)).ToArray();
            foreach (var pipeline in PairGraphicsPipelines(validVertices, validFragments))
            {
                productionContext.AddSource(
                    $"{pipeline.VertexShader.Name}_{pipeline.FragmentShader.Name}.Feather.Graphics.g.cs",
                    SourceText.From(GenerateGraphicsPipeline(pipeline), Encoding.UTF8));
            }
        });
    }

    private static string GenerateKernel(ShaderModel model)
    {
        var nsOpen = string.IsNullOrEmpty(model.Namespace) ? string.Empty : $"namespace {model.Namespace};\n\n";
        var dimension = model.Kind switch
        {
            ShaderKind.Compute1D => "global::Feather.KernelDimension.One",
            ShaderKind.Compute2D => "global::Feather.KernelDimension.Two",
            ShaderKind.Compute3D => "global::Feather.KernelDimension.Three",
            _ => "global::Feather.KernelDimension.One"
        };

        var accessors = GenerateResourceAccessors(model);
        var resources = GenerateResources(model);
        var pushConstants = GeneratePushConstantDescriptors(model.Resources.Items);
        var bind = GenerateBindStatements(model);
        var ir = FeatherIrWriter.ToCSharpByteArray(FeatherIrWriter.WriteModule(model));

        return $$"""
            // <auto-generated/>
            #nullable enable
            {{nsOpen}}partial struct {{model.Name}} : global::Feather.Interop.IGeneratedKernel<{{model.Name}}>
            {
            {{accessors}}
                static global::System.ReadOnlySpan<byte> global::Feather.Interop.IGeneratedKernel<{{model.Name}}>.IR => new byte[] { {{ir}} };

                static global::Feather.Interop.KernelDescriptor global::Feather.Interop.IGeneratedKernel<{{model.Name}}>.Descriptor => new(
                    {{dimension}},
                    new global::Feather.Math.int3({{model.ThreadGroup.X}}, {{model.ThreadGroup.Y}}, {{model.ThreadGroup.Z}}),
                    new global::Feather.Interop.ResourceDescriptor[]
                    {
            {{resources}}
                    },
                    {{pushConstants}},
                    {{model.BoundsCheck.ToString().ToLowerInvariant()}},
                    {{model.AutoDiff.ToString().ToLowerInvariant()}},
                    "{{model.Name}}");

                static void global::Feather.Interop.IGeneratedKernel<{{model.Name}}>.Bind(in {{model.Name}} kernel, global::Feather.GpuKernelCommand command)
                {
            {{bind}}
                }
            }
            """;
    }

    private static string GenerateGpuStruct(GpuStructModel model)
    {
        var nsOpen = string.IsNullOrEmpty(model.Namespace) ? string.Empty : $"namespace {model.Namespace};\n\n";
        var fields = GenerateGpuStructFieldDescriptors(model);
        var packBody = GenerateGpuStructPackBody(model);
        var unpackBody = GenerateGpuStructUnpackBody(model);
        var declaration = model.IsRecord ? "partial record struct" : "partial struct";

        return $$"""
            // <auto-generated/>
            #nullable enable
            {{nsOpen}}{{declaration}} {{model.Name}} : global::Feather.Interop.IGpuStruct<{{model.FullyQualifiedMetadataName}}>
            {
                static global::Feather.Interop.GpuStructLayout global::Feather.Interop.IGpuStruct<{{model.FullyQualifiedMetadataName}}>.Layout => new(
                    "{{model.Name}}",
                    {{model.LayoutName}},
                    new global::Feather.Interop.GpuStructField[]
                    {
            {{fields}}
                    },
                    {{model.SizeInBytes}},
                    {{model.Alignment}});

                static void global::Feather.Interop.IGpuStruct<{{model.FullyQualifiedMetadataName}}>.Pack(global::System.ReadOnlySpan<{{model.FullyQualifiedMetadataName}}> source, global::System.Span<byte> destination)
                {
            {{packBody}}
                }

                static void global::Feather.Interop.IGpuStruct<{{model.FullyQualifiedMetadataName}}>.Unpack(global::System.ReadOnlySpan<byte> source, global::System.Span<{{model.FullyQualifiedMetadataName}}> destination)
                {
            {{unpackBody}}
                }
            }
            """;
    }

    private static string GenerateGpuStructFieldDescriptors(GpuStructModel model)
    {
        if (model.Fields.Items.Count == 0)
        {
            return string.Empty;
        }

        var builder = new StringBuilder();
        foreach (var field in model.Fields.Items)
        {
            builder.Append("            new(\"")
                .Append(field.Name)
                .Append("\", typeof(")
                .Append(field.TypeName)
                .Append("), ")
                .Append(field.Offset)
                .Append(", ")
                .Append(field.SizeInBytes)
                .Append(", ")
                .Append(field.Alignment);
            if (field.ArrayLength > 0)
            {
                builder.Append(", ")
                    .Append(field.ArrayLength)
                    .Append(", ")
                    .Append(field.ArrayStride);
            }

            builder.AppendLine("),");
        }

        return builder.ToString();
    }

    private static string GenerateGpuStructPackBody(GpuStructModel model)
    {
        var builder = new StringBuilder();
        builder.Append("        var __feather_required = checked(source.Length * ")
            .Append(model.SizeInBytes)
            .AppendLine(");");
        builder.AppendLine("        if (destination.Length < __feather_required)");
        builder.AppendLine("        {");
        builder.AppendLine("            throw new global::System.ArgumentException(\"Destination span is too small for the generated GPU struct layout.\", nameof(destination));");
        builder.AppendLine("        }");
        builder.AppendLine();
        builder.AppendLine("        for (var __feather_index = 0; __feather_index < source.Length; __feather_index++)");
        builder.AppendLine("        {");
        builder.Append("            var __feather_destination = destination.Slice(checked(__feather_index * ")
            .Append(model.SizeInBytes)
            .Append("), ")
            .Append(model.SizeInBytes)
            .AppendLine(");");
        builder.AppendLine("            __feather_destination.Clear();");

        if (model.Fields.Items.Count == 0)
        {
            builder.AppendLine("            _ = __feather_destination;");
        }
        else
        {
            builder.AppendLine("            var __feather_value = source[__feather_index];");
            builder.AppendLine("            // Fields are written at generator-computed std430 offsets using each type's EasyGPU field layout.");
            foreach (var field in model.Fields.Items)
            {
                var sourceExpression = "__feather_value." + field.MemberAccessor;
                if (field.IsConstructorParameter)
                {
                    var localName = "__feather_field_" + field.Name;
                    builder.Append("            var ")
                        .Append(localName)
                        .Append(" = ")
                        .Append(sourceExpression)
                        .AppendLine(";");
                    sourceExpression = localName;
                }

                if (field.ArrayLength > 0 && field.ArrayElementTypeName is { Length: > 0 } elementTypeName)
                {
                    builder.Append("            for (var __feather_array_index = 0; __feather_array_index < ")
                        .Append(field.ArrayLength)
                        .AppendLine("; __feather_array_index++)");
                    builder.AppendLine("            {");
                    builder.Append("                global::Feather.Interop.GpuValueLayout<")
                        .Append(elementTypeName)
                        .Append(">.PackValue(in ")
                        .Append(sourceExpression)
                        .Append("[__feather_array_index], __feather_destination.Slice(checked(")
                        .Append(field.Offset)
                        .Append(" + (__feather_array_index * ")
                        .Append(field.ArrayStride)
                        .Append(")), ")
                        .Append(field.ArrayStride)
                        .AppendLine("));");
                    builder.AppendLine("            }");
                }
                else
                {
                    builder.Append("            global::Feather.Interop.GpuValueLayout<")
                        .Append(field.TypeName)
                        .Append(">.PackValue(in ")
                        .Append(sourceExpression)
                        .Append(", __feather_destination.Slice(")
                        .Append(field.Offset)
                        .Append(", ")
                        .Append(field.SizeInBytes)
                        .AppendLine("));");
                }
            }
        }

        builder.AppendLine("        }");
        return builder.ToString();
    }

    private static string GenerateGpuStructUnpackBody(GpuStructModel model)
    {
        var builder = new StringBuilder();
        builder.Append("        var __feather_required = checked(destination.Length * ")
            .Append(model.SizeInBytes)
            .AppendLine(");");
        builder.AppendLine("        if (source.Length < __feather_required)");
        builder.AppendLine("        {");
        builder.AppendLine("            throw new global::System.ArgumentException(\"Source span is too small for the generated GPU struct layout.\", nameof(source));");
        builder.AppendLine("        }");
        builder.AppendLine();
        builder.AppendLine("        for (var __feather_index = 0; __feather_index < destination.Length; __feather_index++)");
        builder.AppendLine("        {");
        builder.Append("            var __feather_source = source.Slice(checked(__feather_index * ")
            .Append(model.SizeInBytes)
            .Append("), ")
            .Append(model.SizeInBytes)
            .AppendLine(");");

        if (model.Fields.Items.Count == 0)
        {
            builder.AppendLine("            _ = __feather_source;");
            builder.AppendLine("            destination[__feather_index] = default;");
        }
        else if (model.Fields.Items.Any(static field => field.IsConstructorParameter))
        {
            builder.AppendLine("            // Positional record properties are constructor-backed, so unpack into locals first.");
            foreach (var field in model.Fields.Items)
            {
                builder.Append("            var __feather_field_")
                    .Append(field.Name)
                    .Append(" = default(")
                    .Append(field.TypeName)
                    .AppendLine(");");
            }

            foreach (var field in model.Fields.Items)
            {
                var target = "__feather_field_" + field.Name;

                if (field.ArrayLength > 0 && field.ArrayElementTypeName is { Length: > 0 } elementTypeName)
                {
                    builder.Append("            for (var __feather_array_index = 0; __feather_array_index < ")
                        .Append(field.ArrayLength)
                        .AppendLine("; __feather_array_index++)");
                    builder.AppendLine("            {");
                    builder.Append("                ")
                        .Append(target)
                        .Append("[__feather_array_index] = global::Feather.Interop.GpuValueLayout<")
                        .Append(elementTypeName)
                        .Append(">.UnpackValue(__feather_source.Slice(checked(")
                        .Append(field.Offset)
                        .Append(" + (__feather_array_index * ")
                        .Append(field.ArrayStride)
                        .Append(")), ")
                        .Append(field.ArrayStride)
                        .AppendLine("));");
                    builder.AppendLine("            }");
                }
                else
                {
                    builder.Append("            ")
                        .Append(target)
                        .Append(" = global::Feather.Interop.GpuValueLayout<")
                        .Append(field.TypeName)
                        .Append(">.UnpackValue(__feather_source.Slice(")
                        .Append(field.Offset)
                        .Append(", ")
                        .Append(field.SizeInBytes)
                        .AppendLine("));");
                }
            }

            builder.Append("            var __feather_value = new ")
                .Append(model.FullyQualifiedMetadataName)
                .Append("(");
            builder.Append(string.Join(", ", model.Fields.Items
                .Where(static field => field.IsConstructorParameter)
                .Select(static field => "__feather_field_" + field.Name)));
            builder.AppendLine(");");

            foreach (var field in model.Fields.Items.Where(static field => !field.IsConstructorParameter))
            {
                builder.Append("            __feather_value.")
                    .Append(field.MemberAccessor)
                    .Append(" = __feather_field_")
                    .Append(field.Name)
                    .AppendLine(";");
            }

            builder.AppendLine("            destination[__feather_index] = __feather_value;");
        }
        else
        {
            builder.AppendLine("            var __feather_value = default(" + model.FullyQualifiedMetadataName + ");");
            builder.AppendLine("            // Unpack mirrors the std430 offsets used by Pack so CPU readback follows the same contract.");
            foreach (var field in model.Fields.Items)
            {
                if (field.ArrayLength > 0 && field.ArrayElementTypeName is { Length: > 0 } elementTypeName)
                {
                    builder.Append("            for (var __feather_array_index = 0; __feather_array_index < ")
                        .Append(field.ArrayLength)
                        .AppendLine("; __feather_array_index++)");
                    builder.AppendLine("            {");
                    builder.Append("                __feather_value.")
                        .Append(field.MemberAccessor)
                        .Append("[__feather_array_index] = global::Feather.Interop.GpuValueLayout<")
                        .Append(elementTypeName)
                        .Append(">.UnpackValue(__feather_source.Slice(checked(")
                        .Append(field.Offset)
                        .Append(" + (__feather_array_index * ")
                        .Append(field.ArrayStride)
                        .Append(")), ")
                        .Append(field.ArrayStride)
                        .AppendLine("));");
                    builder.AppendLine("            }");
                }
                else
                {
                    builder.Append("            __feather_value.")
                        .Append(field.MemberAccessor)
                        .Append(" = global::Feather.Interop.GpuValueLayout<")
                        .Append(field.TypeName)
                        .Append(">.UnpackValue(__feather_source.Slice(")
                        .Append(field.Offset)
                        .Append(", ")
                        .Append(field.SizeInBytes)
                        .AppendLine("));");
                }
            }

            builder.AppendLine("            destination[__feather_index] = __feather_value;");
        }

        builder.AppendLine("        }");
        return builder.ToString();
    }

    private static IEnumerable<GraphicsPipelineModel> PairGraphicsPipelines(IReadOnlyList<ShaderModel> vertexShaders, IReadOnlyList<ShaderModel> fragmentShaders)
    {
        foreach (var vertex in vertexShaders.Where(static shader => shader.Kind == ShaderKind.Vertex))
        {
            var vertexInterface = GetGraphicsInterface(vertex, "global::Feather.IVertexShader<");
            if (vertexInterface is null)
            {
                continue;
            }

            var varyings = vertexInterface.TypeArguments[0].ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            foreach (var fragment in fragmentShaders.Where(static shader => shader.Kind == ShaderKind.Fragment))
            {
                var fragmentInterface = GetGraphicsInterface(fragment, "global::Feather.IFragmentShader<");
                if (fragmentInterface is null || fragmentInterface.TypeArguments.Length == 0)
                {
                    continue;
                }

                var fragmentVaryings = fragmentInterface.TypeArguments[0].ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                if (!StringComparer.Ordinal.Equals(varyings, fragmentVaryings))
                {
                    continue;
                }

                yield return new GraphicsPipelineModel(
                    vertex,
                    fragment,
                    varyings,
                    vertexInterface.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                    fragmentInterface.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat));
            }
        }
    }

    private static INamedTypeSymbol? GetGraphicsInterface(ShaderModel model, string prefix)
        => model.Symbol.AllInterfaces.FirstOrDefault(iface => iface.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat).StartsWith(prefix, StringComparison.Ordinal));

    private static string GenerateGraphicsPipeline(GraphicsPipelineModel pipeline)
    {
        var nsOpen = string.IsNullOrEmpty(pipeline.VertexShader.Namespace) ? string.Empty : $"namespace {pipeline.VertexShader.Namespace};\n\n";
        var accessorSuffix = "__" + pipeline.VertexShader.Name + "_" + pipeline.FragmentShader.Name;
        var vertexAccessors = GenerateResourceAccessors(pipeline.VertexShader, accessorSuffix);
        var fragmentAccessors = GenerateResourceAccessors(pipeline.FragmentShader, accessorSuffix);
        var pipelineResources = GetGraphicsPipelineResources(pipeline);
        var vertexBind = GenerateGraphicsBindStatements(pipeline.VertexShader, "shader", pipelineResources, accessorSuffix);
        var fragmentBind = GenerateGraphicsBindStatements(pipeline.FragmentShader, "shader", pipelineResources, accessorSuffix);
        var resources = GenerateResources(pipelineResources);
        var pushConstants = GeneratePushConstantDescriptors(pipelineResources);
        var vertexIr = FeatherIrWriter.ToCSharpByteArray(FeatherIrWriter.WriteModule(pipeline.VertexShader));
        var fragmentIr = FeatherIrWriter.ToCSharpByteArray(FeatherIrWriter.WriteModule(pipeline.FragmentShader));

        return $$"""
            // <auto-generated/>
            #nullable enable
            {{nsOpen}}partial struct {{pipeline.VertexShader.Name}} : global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>
            {
            {{vertexAccessors}}
                static global::System.ReadOnlySpan<byte> global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.IR => new byte[] { {{vertexIr}} };
                static global::System.ReadOnlySpan<byte> global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.VertexIR => new byte[] { {{vertexIr}} };
                static global::System.ReadOnlySpan<byte> global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.FragmentIR => new byte[] { {{fragmentIr}} };

                static global::Feather.Interop.GraphicsPipelineDescriptor global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.Descriptor => new(
                    new global::Feather.Interop.ResourceDescriptor[]
                    {
            {{resources}}
                    },
                    {{pushConstants}},
                    "{{pipeline.VertexShader.Name}}",
                    "{{pipeline.FragmentShader.Name}}");

                static void global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.BindVertex(in {{pipeline.VertexShader.FullyQualifiedMetadataName}} shader, global::Feather.Graphics.GpuGraphicsCommand command)
                {
            {{vertexBind}}
                }

                static void global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.BindFragment(in {{pipeline.FragmentShader.FullyQualifiedMetadataName}} shader, global::Feather.Graphics.GpuGraphicsCommand command)
                {
                }
            }

            partial struct {{pipeline.FragmentShader.Name}} : global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>
            {
            {{fragmentAccessors}}
                static global::System.ReadOnlySpan<byte> global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.IR => new byte[] { {{vertexIr}} };
                static global::System.ReadOnlySpan<byte> global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.VertexIR => new byte[] { {{vertexIr}} };
                static global::System.ReadOnlySpan<byte> global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.FragmentIR => new byte[] { {{fragmentIr}} };

                static global::Feather.Interop.GraphicsPipelineDescriptor global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.Descriptor => new(
                    new global::Feather.Interop.ResourceDescriptor[]
                    {
            {{resources}}
                    },
                    {{pushConstants}},
                    "{{pipeline.VertexShader.Name}}",
                    "{{pipeline.FragmentShader.Name}}");

                static void global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.BindVertex(in {{pipeline.VertexShader.FullyQualifiedMetadataName}} shader, global::Feather.Graphics.GpuGraphicsCommand command)
                {
                }

                static void global::Feather.Interop.IGeneratedGraphicsPipeline<{{pipeline.VertexShader.FullyQualifiedMetadataName}}, {{pipeline.FragmentShader.FullyQualifiedMetadataName}}, {{pipeline.VaryingsTypeName}}>.BindFragment(in {{pipeline.FragmentShader.FullyQualifiedMetadataName}} shader, global::Feather.Graphics.GpuGraphicsCommand command)
                {
            {{fragmentBind}}
                }
            }
            """;
    }

    private static IReadOnlyList<ResourceModel> GetGraphicsPipelineResources(GraphicsPipelineModel pipeline)
        => pipeline.VertexShader.Resources.Items
            .Concat(pipeline.FragmentShader.Resources.Items)
            .GroupBy(resource => (resource.Binding, resource.Name), resource => resource)
            .Select(group => group.First())
            .OrderBy(resource => resource.Binding)
            .ToArray();

    private static string GenerateGraphicsBindStatements(ShaderModel model, string shaderParameterName, IReadOnlyList<ResourceModel> pipelineResources, string accessorSuffix)
    {
        var builder = new StringBuilder();
        foreach (var resource in model.Resources.Items)
        {
            switch (resource.Kind)
            {
                case ResourceKindModel.Buffer:
                    builder.Append("        command.BindBuffer(").Append(resource.Binding).Append(", ").Append(shaderParameterName).Append(".__feather_buffer_").Append(resource.Name).Append(accessorSuffix).AppendLine("());");
                    break;
                case ResourceKindModel.Texture2D:
                case ResourceKindModel.Texture3D:
                    builder.Append("        command.BindTexture(").Append(resource.Binding).Append(", ").Append(shaderParameterName).Append(".__feather_texture_").Append(resource.Name).Append(accessorSuffix).AppendLine("());");
                    break;
                case ResourceKindModel.Sampler:
                    builder.Append("        command.BindSampler(").Append(resource.Binding).Append(", ").Append(shaderParameterName).Append(".__feather_sampler_").Append(resource.Name).Append(accessorSuffix).AppendLine("());");
                    break;
            }
        }

        builder.Append(GenerateGraphicsPushConstantBinding(model.Resources.Items, pipelineResources, shaderParameterName, accessorSuffix));
        return builder.ToString();
    }

    private static string GenerateResourceAccessors(ShaderModel model, string accessorSuffix = "")
    {
        if (model.Resources.Items.Count == 0)
        {
            return string.Empty;
        }

        var builder = new StringBuilder();
        foreach (var resource in model.Resources.Items)
        {
            builder.AppendLine("    // Generated binding accessors bridge primary-constructor resources to native handles.");
            switch (resource.Kind)
            {
                case ResourceKindModel.Buffer:
                    builder.Append("    private readonly global::Feather.Native.FeBufferHandle __feather_buffer_")
                        .Append(resource.Name)
                        .Append(accessorSuffix)
                        .Append("() => ((global::Feather.Resources.IGpuBufferBinding)")
                        .Append(resource.Name)
                        .AppendLine(").NativeBufferHandle;");
                    break;
                case ResourceKindModel.Texture2D:
                case ResourceKindModel.Texture3D:
                    builder.Append("    private readonly global::Feather.Native.FeTextureHandle __feather_texture_")
                        .Append(resource.Name)
                        .Append(accessorSuffix)
                        .Append("() => ((global::Feather.Resources.IGpuTextureBinding)")
                        .Append(resource.Name)
                        .AppendLine(").NativeTextureHandle;");
                    break;
                case ResourceKindModel.Sampler:
                    builder.Append("    private readonly global::Feather.Native.FeSamplerHandle __feather_sampler_")
                        .Append(resource.Name)
                        .Append(accessorSuffix)
                        .Append("() => ((global::Feather.Resources.IGpuSamplerBinding)")
                        .Append(resource.Name)
                        .AppendLine(").NativeSamplerHandle;");
                    break;
                case ResourceKindModel.PushConstant:
                    builder.Append("    private readonly ")
                        .Append(resource.ElementTypeName)
                        .Append(" __feather_push_constant_")
                        .Append(resource.Name)
                        .Append(accessorSuffix)
                        .Append("() => ")
                        .Append(resource.Name);
                    if (resource.IsUniformWrapper)
                    {
                        builder.Append(".Value");
                    }

                    builder.AppendLine(";");
                    break;
            }
        }

        return builder.ToString();
    }

    private static string GenerateResources(ShaderModel model)
        => GenerateResources(model.Resources.Items);

    private static string GenerateResources(IReadOnlyList<ResourceModel> resources)
    {
        if (resources.Count == 0)
        {
            return string.Empty;
        }

        var builder = new StringBuilder();
        foreach (var resource in resources)
        {
            builder.Append("            new(")
                .Append(resource.Binding)
                .Append(", ")
                .Append(ToRuntimeResourceKind(resource.Kind))
                .Append(", ")
                .Append(ToRuntimeResourceAccess(resource.Access))
                .Append(", typeof(")
                .Append(resource.ElementTypeName)
                .Append("), \"")
                .Append(resource.Name)
                .AppendLine("\"),");
        }

        return builder.ToString();
    }

    private static string GenerateBindStatements(ShaderModel model)
    {
        var builder = new StringBuilder();
        foreach (var resource in model.Resources.Items)
        {
            switch (resource.Kind)
            {
                case ResourceKindModel.Buffer:
                    builder.Append("        command.BindBuffer(").Append(resource.Binding).Append(", kernel.__feather_buffer_").Append(resource.Name).AppendLine("());");
                    break;
                case ResourceKindModel.Texture2D:
                case ResourceKindModel.Texture3D:
                    builder.Append("        command.BindTexture(").Append(resource.Binding).Append(", kernel.__feather_texture_").Append(resource.Name).AppendLine("());");
                    break;
                case ResourceKindModel.Sampler:
                    builder.Append("        command.BindSampler(").Append(resource.Binding).Append(", kernel.__feather_sampler_").Append(resource.Name).AppendLine("());");
                    break;
            }
        }

        builder.Append(GenerateComputePushConstantBinding(model.Resources.Items));
        return builder.ToString();
    }

    private static string GeneratePushConstantDescriptors(IReadOnlyList<ResourceModel> resources)
    {
        var pushConstants = resources.Where(static resource => resource.Kind == ResourceKindModel.PushConstant).ToArray();
        if (pushConstants.Length == 0)
        {
            return "global::System.Array.Empty<global::Feather.Interop.PushConstantDescriptor>()";
        }

        var builder = new StringBuilder();
        builder.AppendLine("new global::Feather.Interop.PushConstantDescriptor[]");
        builder.AppendLine("                    {");

        foreach (var resource in pushConstants)
        {
            var size = PushConstantSizeExpression(resource);
            builder.Append("                        new((uint)(")
                .Append(PushConstantOffsetExpression(pushConstants, resource))
                .Append("), (uint)(")
                .Append(size)
                .Append("), typeof(")
                .Append(resource.ElementTypeName)
                .Append("), \"")
                .Append(resource.Name)
                .AppendLine("\"),");
        }

        builder.Append("                    }");
        return builder.ToString();
    }

    private static string GenerateComputePushConstantBinding(IReadOnlyList<ResourceModel> resources)
    {
        var pushConstants = resources.Where(static resource => resource.Kind == ResourceKindModel.PushConstant).ToArray();
        if (pushConstants.Length == 0)
        {
            return string.Empty;
        }

        var builder = new StringBuilder();
        builder.Append("        var __feather_push_constant_size = ")
            .Append(PushConstantTotalSizeExpression(pushConstants))
            .AppendLine(";");
        builder.AppendLine("        global::System.Span<byte> __feather_push_constants = stackalloc byte[__feather_push_constant_size];");

        var offset = "0";
        for (var i = 0; i < pushConstants.Length; i++)
        {
            var resource = pushConstants[i];
            var size = PushConstantSizeExpression(resource);
            offset = AlignOffsetExpression(offset, PushConstantAlignmentExpression(resource));
            builder.Append("        var __feather_push_constant_value_").Append(i).Append(" = ")
                .Append(PushConstantValueExpression("kernel", resource))
                .AppendLine(";");
            builder.Append("        global::Feather.Interop.GpuValueLayout<")
                .Append(resource.ElementTypeName)
                .Append(">.PackValue(in __feather_push_constant_value_")
                .Append(i)
                .Append(", __feather_push_constants.Slice(")
                .Append(offset)
                .Append(", ")
                .Append(size)
                .AppendLine("));");
            offset = AddOffsetExpression(offset, size);
        }

        builder.AppendLine("        command.SetPushConstants(__feather_push_constants);");
        return builder.ToString();
    }

    private static string GenerateGraphicsPushConstantBinding(IReadOnlyList<ResourceModel> stageResources, IReadOnlyList<ResourceModel> pipelineResources, string shaderParameterName, string accessorSuffix)
    {
        var pushConstants = stageResources.Where(static resource => resource.Kind == ResourceKindModel.PushConstant).ToArray();
        if (pushConstants.Length == 0)
        {
            return string.Empty;
        }

        var pipelinePushConstants = pipelineResources.Where(static resource => resource.Kind == ResourceKindModel.PushConstant).ToArray();
        var totalSize = PushConstantTotalSizeExpression(pipelinePushConstants);
        var builder = new StringBuilder();
        for (var i = 0; i < pushConstants.Length; i++)
        {
            var resource = pushConstants[i];
            var size = PushConstantSizeExpression(resource);
            builder.Append("        var __feather_push_constant_value_").Append(i).Append(" = ")
                .Append(PushConstantValueExpression(shaderParameterName, resource, accessorSuffix))
                .AppendLine(";");
            builder.Append("        global::System.Span<byte> __feather_push_constant_bytes_").Append(i)
                .Append(" = stackalloc byte[")
                .Append(size)
                .AppendLine("];");
            builder.Append("        global::Feather.Interop.GpuValueLayout<")
                .Append(resource.ElementTypeName)
                .Append(">.PackValue(in __feather_push_constant_value_")
                .Append(i)
                .Append(", __feather_push_constant_bytes_")
                .Append(i)
                .AppendLine(");");
            builder.Append("        command.SetPushConstantRange((uint)(")
                .Append(PushConstantOffsetExpression(pipelinePushConstants, resource))
                .Append("), __feather_push_constant_bytes_")
                .Append(i)
                .Append(", (uint)(")
                .Append(totalSize)
                .AppendLine("));");
        }

        return builder.ToString();
    }

    private static string PushConstantValueExpression(string instanceName, ResourceModel resource, string accessorSuffix = "")
        => instanceName + ".__feather_push_constant_" + resource.Name + accessorSuffix + "()";

    private static string PushConstantOffsetExpression(IReadOnlyList<ResourceModel> pushConstants, ResourceModel target)
    {
        var offset = "0";
        foreach (var resource in pushConstants)
        {
            offset = AlignOffsetExpression(offset, PushConstantAlignmentExpression(resource));
            if (resource == target || (resource.Binding == target.Binding && resource.Name == target.Name))
            {
                return offset;
            }

            offset = AddOffsetExpression(offset, PushConstantSizeExpression(resource));
        }

        return "0";
    }

    private static string PushConstantTotalSizeExpression(IReadOnlyList<ResourceModel> pushConstants)
    {
        var offset = "0";
        foreach (var resource in pushConstants)
        {
            offset = AlignOffsetExpression(offset, PushConstantAlignmentExpression(resource));
            offset = AddOffsetExpression(offset, PushConstantSizeExpression(resource));
        }

        return offset;
    }

    private static string PushConstantSizeExpression(ResourceModel resource)
        => "global::Feather.Interop.GpuValueLayout<" + resource.ElementTypeName + ">.FieldSizeInBytes";

    private static string PushConstantAlignmentExpression(ResourceModel resource)
        => "global::Feather.Interop.GpuValueLayout<" + resource.ElementTypeName + ">.Alignment";

    private static string AddOffsetExpression(string offset, string size)
        => offset == "0" ? size : "checked(" + offset + " + " + size + ")";

    private static string AlignOffsetExpression(string offset, string alignment)
        => "global::Feather.Interop.GpuValueLayout.AlignOffset(" + offset + ", " + alignment + ")";

    private static string ToRuntimeResourceKind(ResourceKindModel kind)
        => kind switch
        {
            ResourceKindModel.Buffer => "global::Feather.ResourceKind.Buffer",
            ResourceKindModel.Texture2D => "global::Feather.ResourceKind.Texture2D",
            ResourceKindModel.Texture3D => "global::Feather.ResourceKind.Texture3D",
            ResourceKindModel.Sampler => "global::Feather.ResourceKind.Sampler",
            ResourceKindModel.Uniform => "global::Feather.ResourceKind.Uniform",
            _ => "global::Feather.ResourceKind.PushConstant"
        };

    private static string ToRuntimeResourceAccess(ResourceAccessModel access)
        => access switch
        {
            ResourceAccessModel.Read => "global::Feather.ResourceAccess.Read",
            ResourceAccessModel.Write => "global::Feather.ResourceAccess.Write",
            ResourceAccessModel.ReadWrite => "global::Feather.ResourceAccess.ReadWrite",
            ResourceAccessModel.Sample => "global::Feather.ResourceAccess.Sample",
            _ => "global::Feather.ResourceAccess.Read"
        };
}
