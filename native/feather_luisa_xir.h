#pragma once

#include "feather_typed_ir.h"
#include "feather_typed_ir_lowerer.h"

#include <string>
#include <vector>

#include <luisa/xir/function.h>
#include <luisa/xir/module.h>

namespace Feather::Luisa {

struct BufferLayout {
    uint32_t binding = 0;
    uint32_t feir_type_id = TypedIR::NoIndex;
    const luisa::compute::Type* device_type = nullptr;
};

struct AdParameter {
    uint32_t source_binding = 0;
    uint32_t element_count = 0;
};

struct AdInputs {
    std::string loss_name;
    std::vector<AdParameter> parameters;
};

struct AdGradientLayout {
    uint32_t source_binding = 0;
    uint32_t element_count = 0;
    const luisa::compute::Type* device_type = nullptr;
};

luisa::compute::xir::KernelFunction* LowerToXir(
    const TypedIR::Module& module,
    const TypedIR::LoweringInputs& inputs,
    luisa::compute::xir::Module& xir_module,
    std::vector<BufferLayout>* buffer_layouts,
    const AdInputs* ad_inputs = nullptr,
    std::vector<AdGradientLayout>* ad_gradient_layouts = nullptr,
    std::string* error = nullptr);

} // namespace Feather::Luisa
