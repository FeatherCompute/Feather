#include <Backend/Backend.h>
#include <Runtime/Buffer.h>
#include <Runtime/Context.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using GPU::Backend::Backend;
using GPU::Backend::BindingType;
using GPU::Backend::BufferHandle;
using GPU::Backend::PipelineHandle;
using GPU::Backend::ShaderHandle;
using GPU::Backend::ShaderOptimizationLevel;
using GPU::Runtime::Buffer;
using GPU::Runtime::BufferMode;

constexpr uint32_t kWorkgroupSize = 256;

struct Options {
    std::filesystem::path source_directory;
    std::filesystem::path output_path;
    int warmup_iterations = 8;
    int measured_iterations = 40;
    int compile_samples = 7;
    int dispatches_per_sample = 8;
};

struct Scenario {
    std::string name;
    uint32_t element_count;
    double validation_tolerance;
    double allowed_mismatch_fraction;
};

struct SourceInput {
    Scenario scenario;
    std::string authoring;
    std::filesystem::path path;
    std::string source;
};

struct CompilationMeasurement {
    double cold_inspection_median_ms = 0.0;
    double frontend_median_ms = 0.0;
    double optimizer_median_ms = 0.0;
    double warm_inspection_median_ms = 0.0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    size_t optimized_glsl_bytes = 0;
    size_t optimized_glsl_lines = 0;
};

struct BenchmarkResult {
    std::string scenario;
    std::string authoring;
    std::string level;
    size_t source_bytes = 0;
    size_t source_lines = 0;
    CompilationMeasurement compilation;
    double warm_pipeline_setup_ms = 0.0;
    double dispatch_median_ms = 0.0;
    double dispatch_p95_ms = 0.0;
    double dispatch_mean_ms = 0.0;
    double dispatch_min_ms = 0.0;
    double dispatch_max_ms = 0.0;
    double checksum = 0.0;
    double max_absolute_error = 0.0;
    uint64_t mismatched_values = 0;
    bool used_gpu_timestamps = false;
};

struct LevelInfo {
    const char* name;
    ShaderOptimizationLevel value;
};

constexpr LevelInfo kLevels[] = {
    {"None", ShaderOptimizationLevel::None},
    {"Size", ShaderOptimizationLevel::Size},
    {"Aggressive", ShaderOptimizationLevel::Aggressive},
    {"Ultra", ShaderOptimizationLevel::Ultra},
    {"Extreme", ShaderOptimizationLevel::Extreme},
};

const std::vector<Scenario>& benchmark_scenarios() {
    static const std::vector<Scenario> scenarios = {
        {"fused-mlp", 1u << 20, 0.002, 0.0},
        {"particle-sim", 1u << 19, 0.001, 0.0},
    };
    return scenarios;
}

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double percentile(std::vector<double> samples, double fraction) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    const auto index = static_cast<size_t>(std::ceil(fraction * static_cast<double>(samples.size()))) - 1;
    return samples[std::min(index, samples.size() - 1)];
}

double median(std::vector<double> samples) {
    return percentile(std::move(samples), 0.5);
}

size_t line_count(std::string_view text) {
    return static_cast<size_t>(std::count(text.begin(), text.end(), '\n')) + 1;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Could not open shader source: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::string json_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default: escaped += c; break;
        }
    }
    return escaped;
}

void set_cache_directory(const std::filesystem::path& path) {
#ifdef _WIN32
    _putenv_s("EASYGPU_SHADER_CACHE_DIR", path.string().c_str());
#else
    setenv("EASYGPU_SHADER_CACHE_DIR", path.string().c_str(), 1);
#endif
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto require_value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(option) + " requires a value.");
            }
            return argv[++i];
        };

        if (argument == "--sources") {
            options.source_directory = require_value("--sources");
        } else if (argument == "--output") {
            options.output_path = require_value("--output");
        } else if (argument == "--warmup") {
            options.warmup_iterations = std::stoi(require_value("--warmup"));
        } else if (argument == "--iterations") {
            options.measured_iterations = std::stoi(require_value("--iterations"));
        } else if (argument == "--compile-samples") {
            options.compile_samples = std::stoi(require_value("--compile-samples"));
        } else if (argument == "--dispatches-per-sample") {
            options.dispatches_per_sample = std::stoi(require_value("--dispatches-per-sample"));
        } else if (argument == "--quick") {
            options.warmup_iterations = 2;
            options.measured_iterations = 8;
            options.compile_samples = 3;
            options.dispatches_per_sample = 3;
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }

    if (options.source_directory.empty() || options.output_path.empty()) {
        throw std::invalid_argument("Usage: feather_optimization_benchmark --sources DIR --output FILE [--quick]");
    }
    if (options.warmup_iterations < 0 || options.measured_iterations < 1 || options.compile_samples < 1 ||
        options.dispatches_per_sample < 1) {
        throw std::invalid_argument("Iteration counts must be positive; warmup may be zero.");
    }
    return options;
}

std::vector<SourceInput> load_sources(const std::filesystem::path& directory) {
    std::vector<SourceInput> sources;
    for (const auto& scenario : benchmark_scenarios()) {
        for (const std::string authoring : {"Feather", "Handwritten"}) {
            std::string prefix = authoring == "Feather" ? "feather-" : "handwritten-";
            auto path = directory / (prefix + scenario.name + ".comp");
            sources.push_back({scenario, authoring, path, read_text(path)});
        }
    }
    return sources;
}

CompilationMeasurement measure_compilation(
    Backend& backend,
    const SourceInput& input,
    ShaderOptimizationLevel level,
    const std::filesystem::path& cache_directory,
    int sample_count) {
    std::vector<double> cold;
    std::vector<double> frontend;
    std::vector<double> optimizer;
    std::vector<double> warm;
    std::string optimized_glsl;
    uint64_t hits = 0;
    uint64_t misses = 0;

    GPU::Backend::ShaderDesc descriptor;
    descriptor.type = GPU::Backend::ShaderType::Compute;
    descriptor.sourceCode = input.source;
    descriptor.entryPoint = "main";
    descriptor.optimizationLevel = level;

    for (int sample = 0; sample < sample_count; ++sample) {
        std::filesystem::remove_all(cache_directory);
        backend.ResetShaderCompilationStats();
        const auto start = Clock::now();
        optimized_glsl = backend.GetOptimizedGLSL(descriptor);
        const auto elapsed = Clock::now() - start;
        const auto stats = backend.GetShaderCompilationStats();
        if (optimized_glsl.empty()) {
            throw std::runtime_error("Optimized GLSL inspection returned empty output. Build with SPIRV-Cross enabled.");
        }
        cold.push_back(milliseconds(elapsed));
        frontend.push_back(stats.lastFrontendMilliseconds);
        optimizer.push_back(stats.lastOptimizationMilliseconds);
        hits += stats.diskCacheHits;
        misses += stats.diskCacheMisses;
    }

    for (int sample = 0; sample < sample_count; ++sample) {
        backend.ResetShaderCompilationStats();
        const auto start = Clock::now();
        optimized_glsl = backend.GetOptimizedGLSL(descriptor);
        warm.push_back(milliseconds(Clock::now() - start));
        const auto stats = backend.GetShaderCompilationStats();
        hits += stats.diskCacheHits;
        misses += stats.diskCacheMisses;
    }

    return {
        median(std::move(cold)),
        median(std::move(frontend)),
        median(std::move(optimizer)),
        median(std::move(warm)),
        hits,
        misses,
        optimized_glsl.size(),
        line_count(optimized_glsl),
    };
}

class ExecutableCase {
public:
    ExecutableCase(
        Backend& backend,
        const SourceInput& source_input,
        const LevelInfo& level_info,
        std::shared_ptr<Buffer<float>> input_buffer,
        CompilationMeasurement compilation_measurement)
        : backend_(&backend), source_(&source_input), level_(&level_info), input_(std::move(input_buffer)),
          output_(std::make_unique<Buffer<float>>(source_input.scenario.element_count, BufferMode::Write)),
          compilation_(std::move(compilation_measurement)) {
        GPU::Backend::ShaderDesc shader_descriptor;
        shader_descriptor.type = GPU::Backend::ShaderType::Compute;
        shader_descriptor.sourceCode = source_input.source;
        shader_descriptor.entryPoint = "main";
        shader_descriptor.optimizationLevel = level_info.value;

        const auto setup_start = Clock::now();
        const ShaderHandle shader = backend.CreateShader(shader_descriptor);
        if (shader == GPU::Backend::INVALID_SHADER_HANDLE) {
            throw std::runtime_error("Shader creation failed for " + source_input.scenario.name + "/" +
                                     source_input.authoring + "/" + level_info.name);
        }

        GPU::Backend::PipelineDesc pipeline_descriptor;
        pipeline_descriptor.computeShader = shader;
        pipeline_descriptor.workGroupSizeX = kWorkgroupSize;
        pipeline_descriptor.workGroupSizeY = 1;
        pipeline_descriptor.workGroupSizeZ = 1;
        pipeline_descriptor.resources = {
            {0, BindingType::Buffer, GPU::Backend::PixelFormat::RGBA8, true},
            {1, BindingType::Buffer, GPU::Backend::PixelFormat::RGBA8, false},
        };
        pipeline_ = backend.CreatePipeline(pipeline_descriptor);
        backend.DestroyShader(shader);
        if (pipeline_ == GPU::Backend::INVALID_PIPELINE_HANDLE) {
            throw std::runtime_error("Pipeline creation failed for " + source_input.scenario.name + "/" +
                                     source_input.authoring + "/" + level_info.name);
        }
        setup_ms_ = milliseconds(Clock::now() - setup_start);
    }

    ~ExecutableCase() {
        if (pipeline_ != GPU::Backend::INVALID_PIPELINE_HANDLE && backend_ != nullptr) {
            backend_->DestroyPipeline(pipeline_);
        }
    }

    ExecutableCase(const ExecutableCase&) = delete;
    ExecutableCase& operator=(const ExecutableCase&) = delete;

    void warmup_dispatch(int dispatch_count) {
        bind();
        for (int dispatch = 0; dispatch < dispatch_count; ++dispatch) {
            backend_->Dispatch(group_count(), 1, 1);
            backend_->MemoryBarrier(GPU::Backend::BarrierType::Buffer);
        }
        backend_->Finish();
    }

    void measured_dispatch(int dispatch_count) {
        bind();
        const uint32_t query = backend_->BeginQuery();
        const auto host_start = Clock::now();
        for (int dispatch = 0; dispatch < dispatch_count; ++dispatch) {
            backend_->Dispatch(group_count(), 1, 1);
            backend_->MemoryBarrier(GPU::Backend::BarrierType::Buffer);
        }
        const uint64_t nanoseconds = query == 0 ? 0 : backend_->EndQuery(query);
        double elapsed_ms = 0.0;
        if (nanoseconds > 0) {
            elapsed_ms = static_cast<double>(nanoseconds) / 1'000'000.0;
            used_gpu_timestamps_ = true;
        } else {
            backend_->Finish();
            elapsed_ms = milliseconds(Clock::now() - host_start);
        }
        dispatch_samples_ms_.push_back(elapsed_ms / static_cast<double>(dispatch_count));
    }

    std::vector<float> download() {
        std::vector<float> values(source_->scenario.element_count);
        output_->Download(values.data(), values.size());
        return values;
    }

    BenchmarkResult finish(const std::vector<float>& reference) {
        const auto values = download();
        double checksum = 0.0;
        double max_error = 0.0;
        uint64_t mismatches = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i])) {
                throw std::runtime_error("Non-finite output at index " + std::to_string(i));
            }
            checksum += static_cast<double>(values[i]);
            const double error = std::abs(static_cast<double>(values[i]) - static_cast<double>(reference[i]));
            max_error = std::max(max_error, error);
            if (error > source_->scenario.validation_tolerance) {
                ++mismatches;
            }
        }

        const auto allowed_mismatches = static_cast<uint64_t>(
            std::ceil(static_cast<double>(values.size()) * source_->scenario.allowed_mismatch_fraction));
        if (mismatches > allowed_mismatches) {
            throw std::runtime_error(
                source_->scenario.name + "/" + source_->authoring + "/" + level_->name +
                " failed validation: " + std::to_string(mismatches) + " values exceeded tolerance " +
                std::to_string(source_->scenario.validation_tolerance) + " (allowed " +
                std::to_string(allowed_mismatches) + "), max error " + std::to_string(max_error));
        }

        const double mean = std::accumulate(dispatch_samples_ms_.begin(), dispatch_samples_ms_.end(), 0.0) /
                            static_cast<double>(dispatch_samples_ms_.size());
        return {
            source_->scenario.name,
            source_->authoring,
            level_->name,
            source_->source.size(),
            line_count(source_->source),
            compilation_,
            setup_ms_,
            median(dispatch_samples_ms_),
            percentile(dispatch_samples_ms_, 0.95),
            mean,
            *std::min_element(dispatch_samples_ms_.begin(), dispatch_samples_ms_.end()),
            *std::max_element(dispatch_samples_ms_.begin(), dispatch_samples_ms_.end()),
            checksum,
            max_error,
            mismatches,
            used_gpu_timestamps_,
        };
    }

    const SourceInput& source() const { return *source_; }
    const LevelInfo& level() const { return *level_; }

private:
    uint32_t group_count() const { return source_->scenario.element_count / kWorkgroupSize; }

    void bind() {
        backend_->BindPipeline(pipeline_);
        GPU::Backend::ResourceBinding bindings[2];
        bindings[0].binding = 0;
        bindings[0].type = BindingType::Buffer;
        bindings[0].buffer = input_->GetHandle();
        bindings[0].readOnly = true;
        bindings[1].binding = 1;
        bindings[1].type = BindingType::Buffer;
        bindings[1].buffer = output_->GetHandle();
        bindings[1].readOnly = false;
        backend_->BindResources(bindings, 2);
    }

    Backend* backend_ = nullptr;
    const SourceInput* source_ = nullptr;
    const LevelInfo* level_ = nullptr;
    std::shared_ptr<Buffer<float>> input_;
    std::unique_ptr<Buffer<float>> output_;
    PipelineHandle pipeline_ = GPU::Backend::INVALID_PIPELINE_HANDLE;
    CompilationMeasurement compilation_;
    double setup_ms_ = 0.0;
    bool used_gpu_timestamps_ = false;
    std::vector<double> dispatch_samples_ms_;
};

std::vector<float> make_input(uint32_t count) {
    std::vector<float> values(count);
    uint32_t state = 0x12345678u;
    for (float& value : values) {
        state = state * 1664525u + 1013904223u;
        value = 0.05f + static_cast<float>((state >> 8) & 0x00ffffffu) / 16777216.0f * 0.9f;
    }
    return values;
}

std::vector<float> reference_for_scenario(
    const std::string& scenario,
    std::vector<std::unique_ptr<ExecutableCase>>& cases) {
    for (auto& item : cases) {
        if (item->source().scenario.name == scenario && item->source().authoring == "Handwritten" &&
            std::string_view(item->level().name) == "Ultra") {
            return item->download();
        }
    }
    throw std::runtime_error("Missing Handwritten/Ultra reference for " + scenario);
}

void write_json(
    const Options& options,
    const GPU::Backend::BackendCaps& caps,
    const std::vector<BenchmarkResult>& results) {
    std::filesystem::create_directories(options.output_path.parent_path());
    std::ofstream output(options.output_path);
    if (!output) {
        throw std::runtime_error("Could not write benchmark result: " + options.output_path.string());
    }
    output << std::fixed << std::setprecision(9);
    output << "{\n";
    output << "  \"backend\": \"Vulkan\",\n";
    output << "  \"device\": \"" << json_escape(caps.versionString) << "\",\n";
    output << "  \"timestampQueriesSupported\": " << (caps.supportsTimestampQueries ? "true" : "false") << ",\n";
    output << "  \"warmupIterations\": " << options.warmup_iterations << ",\n";
    output << "  \"measuredIterations\": " << options.measured_iterations << ",\n";
    output << "  \"compileSamples\": " << options.compile_samples << ",\n";
    output << "  \"dispatchesPerSample\": " << options.dispatches_per_sample << ",\n";
    output << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        const auto& compile = result.compilation;
        output << "    {\n";
        output << "      \"scenario\": \"" << json_escape(result.scenario) << "\",\n";
        output << "      \"authoring\": \"" << json_escape(result.authoring) << "\",\n";
        output << "      \"level\": \"" << json_escape(result.level) << "\",\n";
        output << "      \"sourceBytes\": " << result.source_bytes << ",\n";
        output << "      \"sourceLines\": " << result.source_lines << ",\n";
        output << "      \"optimizedGlslBytes\": " << compile.optimized_glsl_bytes << ",\n";
        output << "      \"optimizedGlslLines\": " << compile.optimized_glsl_lines << ",\n";
        output << "      \"coldInspectionMedianMs\": " << compile.cold_inspection_median_ms << ",\n";
        output << "      \"frontendMedianMs\": " << compile.frontend_median_ms << ",\n";
        output << "      \"optimizerMedianMs\": " << compile.optimizer_median_ms << ",\n";
        output << "      \"warmInspectionMedianMs\": " << compile.warm_inspection_median_ms << ",\n";
        output << "      \"cacheHits\": " << compile.cache_hits << ",\n";
        output << "      \"cacheMisses\": " << compile.cache_misses << ",\n";
        output << "      \"warmPipelineSetupMs\": " << result.warm_pipeline_setup_ms << ",\n";
        output << "      \"dispatchMedianMs\": " << result.dispatch_median_ms << ",\n";
        output << "      \"dispatchP95Ms\": " << result.dispatch_p95_ms << ",\n";
        output << "      \"dispatchMeanMs\": " << result.dispatch_mean_ms << ",\n";
        output << "      \"dispatchMinMs\": " << result.dispatch_min_ms << ",\n";
        output << "      \"dispatchMaxMs\": " << result.dispatch_max_ms << ",\n";
        output << "      \"checksum\": " << result.checksum << ",\n";
        output << "      \"maxAbsoluteError\": " << result.max_absolute_error << ",\n";
        output << "      \"mismatchedValues\": " << result.mismatched_values << ",\n";
        output << "      \"usedGpuTimestamps\": " << (result.used_gpu_timestamps ? "true" : "false") << "\n";
        output << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n";
    output << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto cache_directory = options.output_path.parent_path() / "shader-cache";
        std::filesystem::remove_all(cache_directory);
        set_cache_directory(cache_directory);

        const auto sources = load_sources(options.source_directory);
        GPU::Runtime::AutoInitContext();
        GPU::Runtime::ContextGuard guard(GPU::Runtime::Context::GetInstance());
        auto* backend = GPU::Runtime::Context::GetBackend();
        if (backend == nullptr) {
            throw std::runtime_error("EasyGPU backend is unavailable.");
        }
        if (GPU::Runtime::Context::GetInstance().GetBackendType() != GPU::Backend::BackendType::Vulkan) {
            throw std::runtime_error("Optimization benchmark requires the Vulkan backend.");
        }

        std::cout << "Backend: Vulkan (" << backend->GetCaps().versionString << ")\n";
        std::cout << "Preparing " << sources.size() * std::size(kLevels) << " source/level cases...\n";

        std::vector<std::shared_ptr<Buffer<float>>> input_buffers;
        for (const auto& scenario : benchmark_scenarios()) {
            input_buffers.push_back(std::make_shared<Buffer<float>>(make_input(scenario.element_count), BufferMode::Read));
        }

        std::vector<std::unique_ptr<ExecutableCase>> cases;
        cases.reserve(sources.size() * std::size(kLevels));
        for (const auto& source : sources) {
            const auto scenario = std::find_if(
                benchmark_scenarios().begin(), benchmark_scenarios().end(),
                [&](const Scenario& item) { return item.name == source.scenario.name; });
            if (scenario == benchmark_scenarios().end()) {
                throw std::runtime_error("Unknown benchmark scenario: " + source.scenario.name);
            }
            const auto input_index = static_cast<size_t>(std::distance(benchmark_scenarios().begin(), scenario));
            for (const auto& level : kLevels) {
                auto compilation = measure_compilation(
                    *backend, source, level.value, cache_directory, options.compile_samples);
                cases.push_back(std::make_unique<ExecutableCase>(
                    *backend, source, level, input_buffers[input_index], std::move(compilation)));
            }
        }

        for (int round = 0; round < options.warmup_iterations; ++round) {
            for (size_t offset = 0; offset < cases.size(); ++offset) {
                cases[(offset + static_cast<size_t>(round)) % cases.size()]->warmup_dispatch(
                    options.dispatches_per_sample);
            }
        }
        for (int round = 0; round < options.measured_iterations; ++round) {
            for (size_t offset = 0; offset < cases.size(); ++offset) {
                cases[(offset + static_cast<size_t>(round)) % cases.size()]->measured_dispatch(
                    options.dispatches_per_sample);
            }
        }

        const auto fused_reference = reference_for_scenario("fused-mlp", cases);
        const auto particle_reference = reference_for_scenario("particle-sim", cases);
        std::vector<BenchmarkResult> results;
        results.reserve(cases.size());
        for (auto& item : cases) {
            const auto& reference = item->source().scenario.name == "fused-mlp" ? fused_reference : particle_reference;
            auto result = item->finish(reference);
            std::cout << std::left << std::setw(14) << result.scenario << std::setw(13) << result.authoring
                      << std::setw(12) << result.level << std::right << std::setw(10) << std::fixed
                      << std::setprecision(4) << result.dispatch_median_ms << " ms median, " << std::setw(8)
                      << result.compilation.cold_inspection_median_ms << " ms cold\n";
            results.push_back(std::move(result));
        }

        write_json(options, backend->GetCaps(), results);
        std::filesystem::remove_all(cache_directory);
        std::cout << "JSON: " << options.output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "optimization benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
