#pragma once

#include "feather_typed_ir.h"
#include "feather_typed_ir_lowerer.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Feather::Luisa {

struct HostBufferBinding {
    uint32_t binding = 0;
    uint8_t access = 0;
    uint32_t stride = 0;
    std::vector<unsigned char>* bytes = nullptr;
};

struct DispatchInputs {
    uint32_t group_x = 1;
    uint32_t group_y = 1;
    uint32_t group_z = 1;
    uint32_t logical_x = 1;
    uint32_t logical_y = 1;
    uint32_t logical_z = 1;
    std::string runtime_directory;
};

std::string RuntimeDirectory();

bool Dispatch(const TypedIR::Module& module, const TypedIR::LoweringInputs& lowering,
              std::span<HostBufferBinding> buffers, const DispatchInputs& dispatch, std::string* error = nullptr);

} // namespace Feather::Luisa
