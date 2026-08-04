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

luisa::compute::xir::KernelFunction* LowerToXir(
    const TypedIR::Module& module,
    const TypedIR::LoweringInputs& inputs,
    luisa::compute::xir::Module& xir_module,
    std::vector<BufferLayout>* buffer_layouts,
    std::string* error = nullptr);

} // namespace Feather::Luisa
