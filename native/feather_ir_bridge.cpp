#include "feather_c_api.h"
#include "feather_typed_ir.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr uint64_t kMinimumHeaderSize = 44;
constexpr uint64_t kResourceRecordSize = 15;
constexpr uint64_t kInstructionRecordSize = 8;
constexpr uint64_t kSectionRecordSize = 8;
constexpr uint32_t kSectionElementwiseAssignments = 1;
constexpr uint32_t kSectionElementwiseExpressionAssignments = 2;
constexpr uint64_t kAssignmentHeaderSize = 4;
constexpr uint64_t kAssignmentRecordSize = 28;
constexpr uint64_t kExpressionAssignmentHeaderSize = 8;
constexpr uint64_t kExpressionAssignmentHeaderWithArgumentsSize = 12;
constexpr uint64_t kExpressionAssignmentRecordSize = 16;
constexpr uint64_t kExpressionNodeRecordSize = 28;
constexpr uint64_t kExpressionNodeRecordWithArgumentsSize = 40;
constexpr uint64_t kTypedIrHeaderSize = 104;
constexpr uint32_t kSectionControlFlowExpressions = 3;
constexpr uint32_t kSectionAdAnnotations = 4;
constexpr uint32_t kSectionLocalVariables = 5;
constexpr uint32_t kSectionCompoundAssignments = 6;
constexpr uint32_t kSectionTypedShaderIr = 7;
constexpr uint32_t kAdRoleParameter = 0;
constexpr uint32_t kAdRoleLoss = 1;
constexpr uint64_t kAdAnnotationRecordSize = 32;
constexpr uint8_t kOpcodeIf = 4;
constexpr uint8_t kOpcodeBeginBlock = 13;
constexpr uint8_t kOpcodeElse = 14;
constexpr uint8_t kOpcodeEndBlock = 15;
constexpr uint8_t kOpcodeSharedMemoryDeclaration = 28;
constexpr uint8_t kOperandKindSymbol = 3;
constexpr uint8_t kBlockKindGeneric = 0;
constexpr uint8_t kBlockKindIfTrue = 1;
constexpr uint8_t kBlockKindIfElse = 2;

uint16_t read_u16(const unsigned char* data) {
    return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t read_u32(const unsigned char* data) {
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

bool checked_add(uint64_t a, uint64_t b, uint64_t* result) {
    if (UINT64_MAX - a < b) {
        return false;
    }
    *result = a + b;
    return true;
}

bool validate_instruction_structure(const unsigned char* instructions, uint32_t instruction_count) {
    std::vector<uint8_t> block_stack;
    uint8_t pending_block_kind = kBlockKindGeneric;
    bool may_start_else = false;
    for (uint32_t i = 0; i < instruction_count; ++i) {
        const auto* instruction = instructions + (static_cast<uint64_t>(i) * kInstructionRecordSize);
        const auto opcode = instruction[0];
        const auto operand_kind = instruction[1];
        if (opcode == 0 || opcode > kOpcodeSharedMemoryDeclaration || operand_kind > kOperandKindSymbol) {
            return false;
        }

        if (opcode == kOpcodeIf) {
            pending_block_kind = kBlockKindIfTrue;
            may_start_else = false;
            continue;
        }

        // The IR stream is linear, so a tiny block stack preserves enough structure for native validation
        // before the full EasyGPU block/function builder is linked in.
        if (opcode == kOpcodeBeginBlock) {
            block_stack.push_back(pending_block_kind);
            pending_block_kind = kBlockKindGeneric;
            may_start_else = false;
            continue;
        }

        if (opcode == kOpcodeEndBlock) {
            if (block_stack.empty()) {
                return false;
            }

            const auto ended_block = block_stack.back();
            block_stack.pop_back();
            may_start_else = ended_block == kBlockKindIfTrue;
            continue;
        }

        if (opcode == kOpcodeElse) {
            if (!may_start_else) {
                return false;
            }

            pending_block_kind = kBlockKindIfElse;
            may_start_else = false;
            continue;
        }

        pending_block_kind = kBlockKindGeneric;
        may_start_else = false;
    }

    return block_stack.empty();
}

bool validate_typed_shader_ir_section(const unsigned char* payload, uint64_t byte_length) {
    return Feather::TypedIR::ValidateSection(payload, byte_length);
}

uint64_t section_minimum_length(uint32_t kind) {
    switch (kind) {
    case kSectionElementwiseAssignments:
        return kAssignmentHeaderSize;
    case kSectionElementwiseExpressionAssignments:
        return kExpressionAssignmentHeaderSize;
    case kSectionControlFlowExpressions:
        return 12;
    case kSectionAdAnnotations:
        return 8;
    case kSectionLocalVariables:
        return 4;
    case kSectionCompoundAssignments:
        return 12;
    case kSectionTypedShaderIr:
        return kTypedIrHeaderSize;
    default:
        return 0;
    }
}

} // namespace

extern "C" FE_API uint32_t fe_ir_bridge_contract_version() {
    return 1;
}

extern "C" FE_API FeResult fe_ir_validate(const void* ir_data, uint64_t ir_size) {
    if (ir_data == nullptr || ir_size < kMinimumHeaderSize) {
        return FE_ERROR_INVALID_ARGUMENT;
    }

    const auto* data = static_cast<const unsigned char*>(ir_data);
    if (std::memcmp(data, "FEIR", 4) != 0) {
        return FE_ERROR_INVALID_ARGUMENT;
    }

    const auto major = read_u16(data + 4);
    const auto minor = read_u16(data + 6);
    const auto endian = data[8];
    const auto shader_kind = data[9];
    if (major != 1 || endian != 1 || shader_kind == 0 || shader_kind > 5) {
        return FE_ERROR_UNSUPPORTED;
    }

    // IR minor version 1 uses the reserved header slot at byte 10 as a section count.
    // Minor version 0 payloads therefore keep this value at zero and jump straight to strings.
    const auto section_count = read_u16(data + 10);
    const auto resource_count = read_u32(data + 28);
    const auto instruction_count = read_u32(data + 36);
    const auto string_byte_length = read_u32(data + 40);
    uint64_t resource_bytes = 0;
    if (!checked_add(0, static_cast<uint64_t>(resource_count) * kResourceRecordSize, &resource_bytes)) {
        return FE_ERROR_INVALID_ARGUMENT;
    }

    uint64_t instruction_bytes = 0;
    if (!checked_add(0, static_cast<uint64_t>(instruction_count) * kInstructionRecordSize, &instruction_bytes)) {
        return FE_ERROR_INVALID_ARGUMENT;
    }

    uint64_t section_record_bytes = 0;
    if (!checked_add(0, static_cast<uint64_t>(section_count) * kSectionRecordSize, &section_record_bytes)) {
        return FE_ERROR_INVALID_ARGUMENT;
    }

    const auto section_table_offset = kMinimumHeaderSize + resource_bytes + instruction_bytes;
    uint64_t section_payload_bytes = 0;
    if (section_count > 0) {
        if (minor == 0) {
            return FE_ERROR_INVALID_ARGUMENT;
        }

        uint64_t section_table_end = 0;
        if (!checked_add(section_table_offset, section_record_bytes, &section_table_end) || section_table_end > ir_size) {
            return FE_ERROR_INVALID_ARGUMENT;
        }

        for (uint32_t i = 0; i < section_count; ++i) {
            const auto* section = data + section_table_offset + (static_cast<uint64_t>(i) * kSectionRecordSize);
            const auto kind = read_u32(section);
            const auto byte_length = read_u32(section + 4);
            if (kind != kSectionElementwiseAssignments && kind != kSectionElementwiseExpressionAssignments &&
                kind != kSectionControlFlowExpressions &&
                kind != kSectionAdAnnotations &&
                kind != kSectionLocalVariables &&
                kind != kSectionCompoundAssignments &&
                kind != kSectionTypedShaderIr) {
                return FE_ERROR_UNSUPPORTED;
            }

            const auto minimum_length = section_minimum_length(kind);
            if (minimum_length == 0 || byte_length < minimum_length) {
                return FE_ERROR_INVALID_ARGUMENT;
            }

            if (!checked_add(section_payload_bytes, byte_length, &section_payload_bytes)) {
                return FE_ERROR_INVALID_ARGUMENT;
            }
        }
    }

    uint64_t expected = 44;
    if (!checked_add(expected, resource_bytes, &expected) || !checked_add(expected, instruction_bytes, &expected) ||
        !checked_add(expected, section_record_bytes, &expected) ||
        !checked_add(expected, section_payload_bytes, &expected) ||
        !checked_add(expected, string_byte_length, &expected)) {
        return FE_ERROR_INVALID_ARGUMENT;
    }

    if (expected != ir_size) {
        return FE_ERROR_INVALID_ARGUMENT;
    }

    const auto* instruction_data = data + kMinimumHeaderSize + resource_bytes;
    if (!validate_instruction_structure(instruction_data, instruction_count)) {
        return FE_ERROR_INVALID_ARGUMENT;
    }

    uint64_t section_payload_offset = section_table_offset + section_record_bytes;
    for (uint32_t i = 0; i < section_count; ++i) {
        const auto* section = data + section_table_offset + (static_cast<uint64_t>(i) * kSectionRecordSize);
        const auto kind = read_u32(section);
        const auto byte_length = read_u32(section + 4);
        const auto* payload = data + section_payload_offset;
        if (kind == kSectionLocalVariables) {
            if (byte_length < 4) return FE_ERROR_INVALID_ARGUMENT;
            const auto decl_count = read_u32(payload);
            uint64_t expected = 4;
            if (!checked_add(expected, static_cast<uint64_t>(decl_count) * 12, &expected) ||
                expected != byte_length)
                return FE_ERROR_INVALID_ARGUMENT;
            section_payload_offset += byte_length;
            continue;
        }
        if (kind == kSectionCompoundAssignments) {
            if (byte_length < 12) return FE_ERROR_INVALID_ARGUMENT;
            section_payload_offset += byte_length;
            continue;
        }
        if (kind == kSectionAdAnnotations) {
            if (byte_length < 8) return FE_ERROR_INVALID_ARGUMENT;
            const auto maybe_version = read_u16(payload);
            if (maybe_version == 2) {
                if (byte_length < 12) return FE_ERROR_INVALID_ARGUMENT;
                const auto param_count = read_u32(payload + 4);
                const auto loss_count = read_u32(payload + 8);
                uint64_t record_count = static_cast<uint64_t>(param_count) + static_cast<uint64_t>(loss_count);
                uint64_t records_size = 0;
                uint64_t expected = 12;
                if (!checked_add(0, record_count * kAdAnnotationRecordSize, &records_size) ||
                    !checked_add(expected, records_size, &expected) ||
                    expected != byte_length) {
                    return FE_ERROR_INVALID_ARGUMENT;
                }

                const auto* records = payload + 12;
                for (uint32_t record_index = 0; record_index < param_count + loss_count; ++record_index) {
                    const auto* record = records + (static_cast<uint64_t>(record_index) * kAdAnnotationRecordSize);
                    const auto role = read_u32(record);
                    if ((record_index < param_count && role != kAdRoleParameter) ||
                        (record_index >= param_count && role != kAdRoleLoss)) {
                        return FE_ERROR_INVALID_ARGUMENT;
                    }
                }
            } else {
                const auto param_count = read_u32(payload);
                const auto loss_count = read_u32(payload + 4);
                uint64_t expected = 8;
                if (!checked_add(expected, static_cast<uint64_t>(param_count) * 4, &expected) ||
                    !checked_add(expected, static_cast<uint64_t>(loss_count) * 4, &expected) ||
                    expected != byte_length)
                    return FE_ERROR_INVALID_ARGUMENT;
            }
            section_payload_offset += byte_length;
            continue;
        }
        if (kind == kSectionControlFlowExpressions) {
            const auto record_count = read_u32(payload);
            const auto node_count = read_u32(payload + 4);
            const auto argument_index_count = read_u32(payload + 8);
            constexpr uint64_t kCfRecordSize = 12;
            constexpr uint64_t kCfNodeRecordSize = 40;
            uint64_t records_size = 0;
            uint64_t nodes_size = 0;
            uint64_t args_size = 0;
            if (!checked_add(0, static_cast<uint64_t>(record_count) * kCfRecordSize, &records_size) ||
                !checked_add(0, static_cast<uint64_t>(node_count) * kCfNodeRecordSize, &nodes_size) ||
                !checked_add(0, static_cast<uint64_t>(argument_index_count) * sizeof(uint32_t), &args_size)) {
                return FE_ERROR_INVALID_ARGUMENT;
            }
            uint64_t cf_expected = 12;
            if (!checked_add(cf_expected, records_size, &cf_expected) || !checked_add(cf_expected, nodes_size, &cf_expected) ||
                !checked_add(cf_expected, args_size, &cf_expected) || cf_expected != byte_length) {
                return FE_ERROR_INVALID_ARGUMENT;
            }
            const auto* records = payload + 12;
            for (uint32_t rec = 0; rec < record_count; ++rec) {
                const auto* record = records + (static_cast<uint64_t>(rec) * kCfRecordSize);
                if (read_u32(record) >= instruction_count || record[4] == 0 || record[4] > 6 ||
                    read_u32(record + 8) >= node_count) {
                    return FE_ERROR_INVALID_ARGUMENT;
                }
            }
            const auto* cf_nodes = records + records_size;
            for (uint32_t node = 0; node < node_count; ++node) {
                const auto* record = cf_nodes + (static_cast<uint64_t>(node) * kCfNodeRecordSize);
                const auto node_kind = record[0];
                const auto operation = record[1];
                const auto left = read_u32(record + 20);
                const auto right = read_u32(record + 24);
                const auto first_argument = read_u32(record + 32);
                const auto argument_count = read_u32(record + 36);
                if (node_kind == 0 || node_kind > 14 || operation > 14 ||
                    (left != UINT32_MAX && left >= node_count) ||
                    (right != UINT32_MAX && right >= node_count) ||
                    (argument_count > 0 && (first_argument == UINT32_MAX || first_argument > argument_index_count ||
                     argument_count > argument_index_count - first_argument))) {
                    return FE_ERROR_INVALID_ARGUMENT;
                }
            }
            section_payload_offset += byte_length;
            continue;
        }
        if (kind == kSectionElementwiseExpressionAssignments) {
            const auto assignment_count = read_u32(payload);
            const auto node_count = read_u32(payload + 4);
            uint64_t legacy_section_bytes = kExpressionAssignmentHeaderSize;
            uint64_t expression_section_bytes = kExpressionAssignmentHeaderWithArgumentsSize;
            uint64_t assignment_bytes = 0;
            uint64_t node_bytes = 0;
            uint64_t expression_node_bytes = 0;
            uint64_t argument_index_bytes = 0;
            if (!checked_add(0, static_cast<uint64_t>(assignment_count) * kExpressionAssignmentRecordSize, &assignment_bytes) ||
                !checked_add(0, static_cast<uint64_t>(node_count) * kExpressionNodeRecordSize, &node_bytes) ||
                !checked_add(legacy_section_bytes, assignment_bytes, &legacy_section_bytes) ||
                !checked_add(legacy_section_bytes, node_bytes, &legacy_section_bytes)) {
                return FE_ERROR_INVALID_ARGUMENT;
            }

            const auto has_argument_table = legacy_section_bytes != byte_length;
            uint32_t argument_index_count = 0;
            if (has_argument_table) {
                argument_index_count = read_u32(payload + 8);
                if (!checked_add(0, static_cast<uint64_t>(node_count) * kExpressionNodeRecordWithArgumentsSize, &expression_node_bytes) ||
                    !checked_add(0, static_cast<uint64_t>(argument_index_count) * sizeof(uint32_t), &argument_index_bytes) ||
                    !checked_add(expression_section_bytes, assignment_bytes, &expression_section_bytes) ||
                    !checked_add(expression_section_bytes, expression_node_bytes, &expression_section_bytes) ||
                    !checked_add(expression_section_bytes, argument_index_bytes, &expression_section_bytes) ||
                    expression_section_bytes != byte_length) {
                    return FE_ERROR_INVALID_ARGUMENT;
                }
            }

            const auto header_size = has_argument_table ? kExpressionAssignmentHeaderWithArgumentsSize
                                                        : kExpressionAssignmentHeaderSize;
            const auto node_record_size = has_argument_table ? kExpressionNodeRecordWithArgumentsSize
                                                             : kExpressionNodeRecordSize;
            const auto* assignments = payload + header_size;
            const auto* nodes = assignments + assignment_bytes;
            for (uint32_t assignment = 0; assignment < assignment_count; ++assignment) {
                const auto* record = assignments + (static_cast<uint64_t>(assignment) * kExpressionAssignmentRecordSize);
                if (read_u32(record) >= instruction_count || read_u32(record + 12) >= node_count) {
                    return FE_ERROR_INVALID_ARGUMENT;
                }
            }

            for (uint32_t node = 0; node < node_count; ++node) {
                const auto* record = nodes + (static_cast<uint64_t>(node) * node_record_size);
                const auto node_kind = record[0];
                const auto operation = record[1];
                const auto left = read_u32(record + 20);
                const auto right = read_u32(record + 24);
                const auto first_argument = has_argument_table ? read_u32(record + 32) : UINT32_MAX;
                const auto argument_count = has_argument_table ? read_u32(record + 36) : 0;
                if (node_kind == 0 || node_kind > 14 || operation > 14 ||
                    (left != UINT32_MAX && left >= node_count) ||
                    (right != UINT32_MAX && right >= node_count) ||
                    (argument_count > 0 &&
                     (first_argument == UINT32_MAX || first_argument > argument_index_count ||
                      argument_count > argument_index_count - first_argument))) {
                    return FE_ERROR_INVALID_ARGUMENT;
                }
            }

            const auto* argument_indices = nodes + (static_cast<uint64_t>(node_count) * node_record_size);
            for (uint32_t argument = 0; argument < argument_index_count; ++argument) {
                if (read_u32(argument_indices + (static_cast<uint64_t>(argument) * sizeof(uint32_t))) >= node_count) {
                    return FE_ERROR_INVALID_ARGUMENT;
                }
            }

            section_payload_offset += byte_length;
            continue;
        }
        if (kind == kSectionTypedShaderIr) {
            if (!validate_typed_shader_ir_section(payload, byte_length)) {
                return FE_ERROR_INVALID_ARGUMENT;
            }

            section_payload_offset += byte_length;
            continue;
        }

        const auto count = read_u32(payload);
        uint64_t records_size = 0;
        if (!checked_add(0, static_cast<uint64_t>(count) * kAssignmentRecordSize, &records_size) ||
            records_size + kAssignmentHeaderSize != byte_length) {
            return FE_ERROR_INVALID_ARGUMENT;
        }

        for (uint32_t assignment = 0; assignment < count; ++assignment) {
            const auto* record = payload + kAssignmentHeaderSize + (static_cast<uint64_t>(assignment) * kAssignmentRecordSize);
            const auto instruction_index = read_u32(record);
            const auto operation = record[16];
            const auto operand_kind = record[17];
            if (instruction_index >= instruction_count || operation == 0 || operation > 5 || operand_kind > 2) {
                return FE_ERROR_INVALID_ARGUMENT;
            }
        }

        section_payload_offset += byte_length;
    }

    return FE_OK;
}
