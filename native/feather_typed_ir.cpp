#include "feather_typed_ir.h"

#include <cstring>

namespace Feather::TypedIR {
namespace {

constexpr uint64_t kHeaderSize = 104;
constexpr uint64_t kFunctionRecordSize = 25;
constexpr uint64_t kTypeRecordSize = 17;
constexpr uint64_t kStructRecordSize = 24;
constexpr uint64_t kStructFieldRecordSizeV0 = 16;
constexpr uint64_t kStructFieldRecordSizeV1 = 20;
constexpr uint64_t kStatementRecordSize = 29;
constexpr uint64_t kExpressionRecordSize = 33;
constexpr uint64_t kLValueRecordSize = 21;
constexpr uint64_t kParameterRecordSize = 9;

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

Range read_range(const unsigned char* payload, uint64_t offset) {
    return Range{read_u32(payload + offset), read_u32(payload + offset + 4)};
}

bool validate_table_range(Range range, uint64_t record_size, uint64_t expected_offset, uint64_t limit,
                          uint64_t* out_end) {
    uint64_t bytes = 0;
    if (!checked_add(0, static_cast<uint64_t>(range.count) * record_size, &bytes)) {
        return false;
    }

    uint64_t end = 0;
    if (range.offset != expected_offset || !checked_add(range.offset, bytes, &end) || end > limit) {
        return false;
    }

    *out_end = end;
    return true;
}

bool bounded_record_count(Range range, uint64_t record_size, uint64_t byte_length) {
    if (record_size == 0) {
        return false;
    }

    return static_cast<uint64_t>(range.count) <= byte_length / record_size;
}

bool validate_span(uint32_t first, uint32_t count, uint32_t total) {
    return count == 0 ? first == NoIndex : first != NoIndex && first <= total && count <= total - first;
}

bool is_valid_ref(uint32_t index, uint32_t count) {
    return index == NoIndex || index < count;
}

bool validate_texture_sample_expression(uint32_t op, uint32_t argument_count) {
    switch (op) {
    case 0:
        return argument_count == 3;
    case 1:
        return argument_count == 4;
    case 2:
        return argument_count == 5;
    default:
        return false;
    }
}

bool checked_mul_u32(uint32_t a, uint32_t b, uint32_t* result) {
    if (a != 0 && b > UINT32_MAX / a) {
        return false;
    }

    *result = a * b;
    return true;
}

bool align_u32(uint32_t value, uint32_t alignment, uint32_t* result) {
    if (alignment == 0) {
        return false;
    }

    const auto remainder = value % alignment;
    if (remainder == 0) {
        *result = value;
        return true;
    }

    const auto delta = alignment - remainder;
    if (value > UINT32_MAX - delta) {
        return false;
    }

    *result = value + delta;
    return true;
}

bool parse_string_table(const unsigned char* data, uint32_t offset, uint32_t length, uint64_t limit,
                        std::vector<std::string>* strings) {
    uint64_t end = 0;
    if (strings == nullptr || length < sizeof(uint32_t) || !checked_add(offset, length, &end) || end != limit) {
        return false;
    }

    const auto count = read_u32(data + offset);
    uint64_t cursor = static_cast<uint64_t>(offset) + sizeof(uint32_t);
    if (count > (length - sizeof(uint32_t)) / sizeof(uint32_t)) {
        return false;
    }

    strings->clear();
    strings->reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (cursor + sizeof(uint32_t) > end) {
            return false;
        }

    const auto string_length = read_u32(data + cursor);
        cursor += sizeof(uint32_t);
        uint64_t value_end = 0;
        if (!checked_add(cursor, string_length, &value_end) || value_end > end) {
            return false;
        }

        strings->emplace_back(reinterpret_cast<const char*>(data + cursor), string_length);
        cursor = value_end;
    }

    return cursor == end;
}

template <typename T, typename Read>
bool read_records(const unsigned char* payload, Range range, uint64_t record_size, uint64_t byte_length,
                  std::vector<T>* output, Read read) {
    if (output == nullptr || !bounded_record_count(range, record_size, byte_length)) {
        return false;
    }

    output->clear();
    output->reserve(range.count);
    for (uint32_t i = 0; i < range.count; ++i) {
        output->push_back(read(payload + range.offset + (static_cast<uint64_t>(i) * record_size)));
    }

    return true;
}

bool compute_type_layout(const Module& module, uint32_t type_id, std::vector<uint8_t>* visiting,
                         uint32_t* size, uint32_t* alignment);

bool compute_struct_layout(const Module& module, uint32_t struct_index, std::vector<uint8_t>* visiting,
                           uint32_t* size, uint32_t* alignment) {
    if (struct_index >= module.structs.size()) {
        return false;
    }

    const auto& structure = module.structs[struct_index];
    if (structure.alignment == 0 || structure.size_in_bytes % structure.alignment != 0 ||
        !validate_span(structure.first_field, structure.field_count, static_cast<uint32_t>(module.struct_fields.size()))) {
        return false;
    }

    uint32_t cursor = 0;
    uint32_t max_alignment = structure.field_count == 0 ? 1u : 0u;
    for (uint32_t i = 0; i < structure.field_count; ++i) {
        const auto& field = module.struct_fields[structure.first_field + i];
        uint32_t field_size = 0;
        uint32_t field_alignment = 0;
        if (!compute_type_layout(module, field.type_id, visiting, &field_size, &field_alignment) ||
            field.name_id >= module.strings.size() || field.size_in_bytes != field_size) {
            return false;
        }

        uint32_t aligned_cursor = 0;
        if (!align_u32(cursor, field_alignment, &aligned_cursor) ||
            field.offset != aligned_cursor ||
            field.offset > UINT32_MAX - field.size_in_bytes) {
            return false;
        }

        cursor = field.offset + field.size_in_bytes;
        if (field_alignment > max_alignment) {
            max_alignment = field_alignment;
        }
    }

    uint32_t computed_size = 0;
    if (!align_u32(cursor, max_alignment, &computed_size) ||
        structure.size_in_bytes != computed_size ||
        structure.alignment != max_alignment) {
        return false;
    }

    *size = structure.size_in_bytes;
    *alignment = structure.alignment;
    return true;
}

bool compute_type_layout(const Module& module, uint32_t type_id, std::vector<uint8_t>* visiting,
                         uint32_t* size, uint32_t* alignment) {
    if (type_id >= module.types.size() || visiting == nullptr || size == nullptr || alignment == nullptr) {
        return false;
    }

    const auto& type = module.types[type_id];
    switch (type.kind) {
    case 1: // primitive
        if (type.a > 3 || (type.b != 8 && type.b != 16 && type.b != 32)) {
            return false;
        }
        if (type.a == 0 && type.b != 32) {
            return false;
        }
        *size = type.b / 8;
        *alignment = *size;
        return *size != 0;
    case 2: { // vector
        uint32_t element_size = 0;
        uint32_t element_alignment = 0;
        if (type.a >= module.types.size() || type.b < 2 || type.b > 4 ||
            !compute_type_layout(module, type.a, visiting, &element_size, &element_alignment) ||
            element_size != 4 || element_alignment != 4) {
            return false;
        }

        if (!checked_mul_u32(element_size, type.b, size)) {
            return false;
        }
        *alignment = type.b == 2 ? 8u : 16u;
        return true;
    }
    case 3: // matrix
        if (type.a >= module.types.size() || type.b < 2 || type.b > 4 || type.c < 2 || type.c > 4) {
            return false;
        }
        if (module.types[type.a].kind != 1 || module.types[type.a].a != 3 || module.types[type.a].b != 32) {
            return false;
        }
        if (type.b != type.c) {
            return false;
        }
        *size = type.b * 16u;
        *alignment = 16u;
        return true;
    case 4: { // struct
        if (type.a >= module.structs.size()) {
            return false;
        }
        if (type_id >= visiting->size()) {
            return false;
        }
        if ((*visiting)[type_id] != 0) {
            return false;
        }
        (*visiting)[type_id] = 1;
        const auto ok = compute_struct_layout(module, type.a, visiting, size, alignment);
        (*visiting)[type_id] = 0;
        return ok;
    }
    case 5: { // fixed array
        if (type.a >= module.types.size() || type.b == NoIndex || type.b == 0) {
            return false;
        }
        uint32_t element_size = 0;
        uint32_t element_alignment = 0;
        if (!compute_type_layout(module, type.a, visiting, &element_size, &element_alignment)) {
            return false;
        }
        uint32_t stride = 0;
        if (!align_u32(element_size, element_alignment, &stride) ||
            !checked_mul_u32(stride, type.b, size)) {
            return false;
        }
        *alignment = element_alignment;
        return true;
    }
    case 7: // void
        *size = 0;
        *alignment = 1;
        return true;
    default:
        return false;
    }
}

bool validate_struct_layouts(const Module& module) {
    std::vector<uint8_t> visiting(module.types.size(), 0);
    for (uint32_t i = 0; i < module.structs.size(); ++i) {
        uint32_t size = 0;
        uint32_t alignment = 0;
        if (!compute_struct_layout(module, i, &visiting, &size, &alignment)) {
            return false;
        }
    }

    return true;
}

} // namespace

bool ParseSection(const unsigned char* payload, uint64_t byte_length, Module* module) {
    if (module == nullptr || byte_length < kHeaderSize || std::memcmp(payload, "FTIR", 4) != 0) {
        return false;
    }

    const auto major = read_u16(payload + 4);
    const auto minor = read_u16(payload + 6);
    const auto endian = payload[8];
    const auto header_size = read_u16(payload + 10);
    if (major != 1 || minor > 1 || endian != 1 || header_size != kHeaderSize) {
        return false;
    }
    const auto struct_field_record_size = minor >= 1 ? kStructFieldRecordSizeV1 : kStructFieldRecordSizeV0;

    const auto function_range = read_range(payload, 16);
    const auto type_range = read_range(payload, 24);
    const auto struct_range = read_range(payload, 32);
    const auto struct_field_range = read_range(payload, 40);
    const auto statement_range = read_range(payload, 48);
    const auto expression_range = read_range(payload, 56);
    const auto lvalue_range = read_range(payload, 64);
    const auto child_range = read_range(payload, 72);
    const auto argument_range = read_range(payload, 80);
    const auto parameter_range = read_range(payload, 88);
    const auto string_offset = read_u32(payload + 96);
    const auto string_length = read_u32(payload + 100);

    uint64_t next = kHeaderSize;
    if (!validate_table_range(function_range, kFunctionRecordSize, next, byte_length, &next) ||
        !validate_table_range(type_range, kTypeRecordSize, next, byte_length, &next) ||
        !validate_table_range(struct_range, kStructRecordSize, next, byte_length, &next) ||
        !validate_table_range(struct_field_range, struct_field_record_size, next, byte_length, &next) ||
        !validate_table_range(statement_range, kStatementRecordSize, next, byte_length, &next) ||
        !validate_table_range(expression_range, kExpressionRecordSize, next, byte_length, &next) ||
        !validate_table_range(lvalue_range, kLValueRecordSize, next, byte_length, &next) ||
        !validate_table_range(child_range, sizeof(uint32_t), next, byte_length, &next) ||
        !validate_table_range(argument_range, sizeof(uint32_t), next, byte_length, &next) ||
        !validate_table_range(parameter_range, kParameterRecordSize, next, byte_length, &next) ||
        string_offset != next) {
        return false;
    }

    Module parsed;
    parsed.entry_function = read_u32(payload + 12);
    if (!parse_string_table(payload, string_offset, string_length, byte_length, &parsed.strings)) {
        return false;
    }

    if (function_range.count == 0 || parsed.entry_function >= function_range.count ||
        type_range.count == 0 || statement_range.count == 0) {
        return false;
    }

    if (!read_records<Function>(payload, function_range, kFunctionRecordSize, byte_length, &parsed.functions, [](const unsigned char* record) {
        Function function;
        function.kind = record[0];
        function.name_id = read_u32(record + 1);
        function.mangled_name_id = read_u32(record + 5);
        function.return_type_id = read_u32(record + 9);
        function.first_parameter = read_u32(record + 13);
        function.parameter_count = read_u32(record + 17);
        function.body_statement_index = read_u32(record + 21);
        return function;
    })) {
        return false;
    }

    if (!read_records<Type>(payload, type_range, kTypeRecordSize, byte_length, &parsed.types, [](const unsigned char* record) {
        return Type{record[0], read_u32(record + 1), read_u32(record + 5), read_u32(record + 9), read_u32(record + 13)};
    })) {
        return false;
    }

    if (!read_records<StructRecord>(payload, struct_range, kStructRecordSize, byte_length, &parsed.structs, [](const unsigned char* record) {
        return StructRecord{
            read_u32(record),
            read_u32(record + 4),
            read_u32(record + 8),
            read_u32(record + 12),
            read_u32(record + 16),
            read_u32(record + 20)};
    })) {
        return false;
    }

    if (!read_records<StructField>(payload, struct_field_range, struct_field_record_size, byte_length, &parsed.struct_fields, [struct_field_record_size](const unsigned char* record) {
        return StructField{
            read_u32(record),
            read_u32(record + 4),
            read_u32(record + 8),
            read_u32(record + 12),
            struct_field_record_size >= kStructFieldRecordSizeV1 ? read_u32(record + 16) : 0u};
    })) {
        return false;
    }

    if (!read_records<Statement>(payload, statement_range, kStatementRecordSize, byte_length, &parsed.statements, [](const unsigned char* record) {
        Statement statement;
        statement.kind = record[0];
        statement.a = read_u32(record + 1);
        statement.b = read_u32(record + 5);
        statement.c = read_u32(record + 9);
        statement.op = read_u32(record + 13);
        statement.name_id = read_u32(record + 17);
        statement.first_child = read_u32(record + 21);
        statement.child_count = read_u32(record + 25);
        return statement;
    })) {
        return false;
    }

    if (!read_records<Expression>(payload, expression_range, kExpressionRecordSize, byte_length, &parsed.expressions, [](const unsigned char* record) {
        Expression expression;
        expression.kind = record[0];
        expression.type_id = read_u32(record + 1);
        expression.a = read_u32(record + 5);
        expression.b = read_u32(record + 9);
        expression.c = read_u32(record + 13);
        expression.name_id = read_u32(record + 17);
        expression.op = read_u32(record + 21);
        expression.first_argument = read_u32(record + 25);
        expression.argument_count = read_u32(record + 29);
        return expression;
    })) {
        return false;
    }

    if (!read_records<LValue>(payload, lvalue_range, kLValueRecordSize, byte_length, &parsed.lvalues, [](const unsigned char* record) {
        LValue lvalue;
        lvalue.kind = record[0];
        lvalue.type_id = read_u32(record + 1);
        lvalue.a = read_u32(record + 5);
        lvalue.b = read_u32(record + 9);
        lvalue.c = read_u32(record + 13);
        lvalue.name_id = read_u32(record + 17);
        return lvalue;
    })) {
        return false;
    }

    if (!read_records<Parameter>(payload, parameter_range, kParameterRecordSize, byte_length, &parsed.parameters, [](const unsigned char* record) {
        return Parameter{record[0], read_u32(record + 1), read_u32(record + 5)};
    })) {
        return false;
    }

    if (!bounded_record_count(child_range, sizeof(uint32_t), byte_length) ||
        !bounded_record_count(argument_range, sizeof(uint32_t), byte_length)) {
        return false;
    }

    parsed.children.reserve(child_range.count);
    for (uint32_t i = 0; i < child_range.count; ++i) {
        parsed.children.push_back(read_u32(payload + child_range.offset + (static_cast<uint64_t>(i) * sizeof(uint32_t))));
    }

    parsed.arguments.reserve(argument_range.count);
    for (uint32_t i = 0; i < argument_range.count; ++i) {
        parsed.arguments.push_back(read_u32(payload + argument_range.offset + (static_cast<uint64_t>(i) * sizeof(uint32_t))));
    }

    for (uint32_t i = 0; i < parsed.functions.size(); ++i) {
        const auto& function = parsed.functions[i];
        if (function.kind > 5 || function.name_id >= parsed.strings.size() ||
            function.mangled_name_id >= parsed.strings.size() ||
            function.return_type_id >= parsed.types.size() ||
            !validate_span(function.first_parameter, function.parameter_count, static_cast<uint32_t>(parsed.parameters.size())) ||
            function.body_statement_index >= parsed.statements.size()) {
            return false;
        }

        if (function.kind == 5) {
            Callable callable;
            callable.name = parsed.strings[function.mangled_name_id];
            callable.function_index = i;
            parsed.callables[callable.name] = callable;
        }
    }

    for (const auto& type : parsed.types) {
        if (type.kind == 0 || type.kind > 7 ||
            (type.kind == 1 && (type.a > 3 || (type.b != 8 && type.b != 16 && type.b != 32))) ||
            (type.kind == 2 && (type.a >= parsed.types.size() || type.b < 2 || type.b > 4)) ||
            (type.kind == 3 && (type.a >= parsed.types.size() || type.b < 2 || type.b > 4 || type.c < 2 || type.c > 4)) ||
            (type.kind == 4 && type.a >= parsed.structs.size()) ||
            (type.kind == 5 && type.a >= parsed.types.size()) ||
            (type.kind == 6 && (type.a > 3 || type.b >= parsed.types.size() || type.c > 3))) {
            return false;
        }
    }

    for (const auto& structure : parsed.structs) {
        if (structure.name_id >= parsed.strings.size() || structure.fully_qualified_name_id >= parsed.strings.size() ||
            !validate_span(structure.first_field, structure.field_count, static_cast<uint32_t>(parsed.struct_fields.size())) ||
            structure.alignment == 0 || structure.size_in_bytes % structure.alignment != 0) {
            return false;
        }
    }

    for (const auto& field : parsed.struct_fields) {
        if (field.name_id >= parsed.strings.size() || field.type_id >= parsed.types.size() || field.size_in_bytes == 0) {
            return false;
        }
    }

    for (const auto& statement : parsed.statements) {
        if (statement.kind == 0 || statement.kind > 15 ||
            !validate_span(statement.first_child, statement.child_count, static_cast<uint32_t>(parsed.children.size())) ||
            !is_valid_ref(statement.name_id, static_cast<uint32_t>(parsed.strings.size())) ||
            (statement.kind == 2 && (!is_valid_ref(statement.a, static_cast<uint32_t>(parsed.expressions.size())) ||
                                     statement.op >= parsed.types.size())) ||
            (statement.kind == 3 && (statement.a >= parsed.lvalues.size() || statement.b >= parsed.expressions.size())) ||
            (statement.kind == 4 && (statement.a >= parsed.lvalues.size() || statement.b >= parsed.expressions.size() ||
                                     statement.op > 9)) ||
            (statement.kind == 5 && (statement.a >= parsed.expressions.size() || statement.b >= parsed.statements.size() ||
                                     !is_valid_ref(statement.c, static_cast<uint32_t>(parsed.statements.size())))) ||
            (statement.kind == 6 && (!is_valid_ref(statement.a, static_cast<uint32_t>(parsed.statements.size())) ||
                                     !is_valid_ref(statement.b, static_cast<uint32_t>(parsed.expressions.size())) ||
                                     !is_valid_ref(statement.c, static_cast<uint32_t>(parsed.statements.size())) ||
                                     statement.op >= parsed.statements.size())) ||
            (statement.kind == 7 && (statement.a >= parsed.expressions.size() || statement.b >= parsed.statements.size())) ||
            (statement.kind == 8 && (statement.a >= parsed.statements.size() || statement.b >= parsed.expressions.size())) ||
            (statement.kind == 11 && !is_valid_ref(statement.a, static_cast<uint32_t>(parsed.expressions.size()))) ||
            (statement.kind == 12 && statement.a >= parsed.expressions.size()) ||
            (statement.kind == 13 && statement.op > 2) ||
            (statement.kind == 14 && (statement.a >= parsed.lvalues.size() || statement.op > 3)) ||
            (statement.kind == 15 && (statement.a == 0 || statement.op >= parsed.types.size() ||
                                      statement.name_id >= parsed.strings.size()))) {
            return false;
        }

        if (statement.kind != 1 && (statement.first_child != NoIndex || statement.child_count != 0)) {
            return false;
        }
    }

    for (const auto& expression : parsed.expressions) {
        if (expression.kind == 0 || expression.kind > 23 || expression.type_id >= parsed.types.size() ||
            !is_valid_ref(expression.name_id, static_cast<uint32_t>(parsed.strings.size())) ||
            !validate_span(expression.first_argument, expression.argument_count, static_cast<uint32_t>(parsed.arguments.size())) ||
            (expression.kind == 4 && expression.a >= parsed.expressions.size()) ||
            (expression.kind == 5 && expression.a >= parsed.expressions.size()) ||
            (expression.kind == 6 && (expression.a >= parsed.expressions.size() || expression.op > 2)) ||
            (expression.kind == 7 && (expression.a >= parsed.expressions.size() || expression.b >= parsed.expressions.size() ||
                                      expression.op > 9)) ||
            (expression.kind == 8 && (expression.a >= parsed.expressions.size() || expression.b >= parsed.expressions.size() ||
                                      expression.op > 5)) ||
            (expression.kind == 9 && (expression.a >= parsed.expressions.size() || expression.b >= parsed.expressions.size() ||
                                      expression.op > 1)) ||
            (expression.kind == 10 && (expression.a >= parsed.expressions.size() || expression.b >= parsed.expressions.size() ||
                                       expression.c >= parsed.expressions.size())) ||
            (expression.kind == 11 && expression.a >= parsed.expressions.size()) ||
            (expression.kind == 15 && expression.a >= parsed.expressions.size()) ||
            (expression.kind == 16 && expression.a >= parsed.expressions.size()) ||
            (expression.kind == 17 && (expression.a >= parsed.expressions.size() || expression.b >= parsed.expressions.size())) ||
            (expression.kind == 18 && (expression.op == 0 || expression.op > 21)) ||
            (expression.kind == 20 && (expression.a >= parsed.expressions.size() || expression.b >= parsed.expressions.size())) ||
            (expression.kind == 21 && (expression.a >= parsed.expressions.size() ||
                                       expression.name_id >= parsed.strings.size())) ||
            (expression.kind == 22 && (expression.a >= parsed.lvalues.size() || expression.op > 8 ||
                                       expression.argument_count != (expression.op == 8 ? 2u : 1u))) ||
            (expression.kind == 23 && !validate_texture_sample_expression(expression.op, expression.argument_count))) {
            return false;
        }

        const auto is_call_like = expression.kind == 12 || expression.kind == 13 || expression.kind == 14 ||
                                  expression.kind == 22 || expression.kind == 23;
        if (!is_call_like && (expression.argument_count > 0 || expression.first_argument != NoIndex)) {
            return false;
        }
        if (expression.kind == 14) {
            if (expression.name_id >= parsed.strings.size()) {
                return false;
            }

            const auto callable = parsed.callables.find(parsed.strings[expression.name_id]);
            if (callable == parsed.callables.end()) {
                return false;
            }
        }
    }

    for (const auto& lvalue : parsed.lvalues) {
        if (lvalue.kind == 0 || lvalue.kind > 9 || lvalue.type_id >= parsed.types.size() ||
            !is_valid_ref(lvalue.name_id, static_cast<uint32_t>(parsed.strings.size())) ||
            (lvalue.kind == 3 && !is_valid_ref(lvalue.a, static_cast<uint32_t>(parsed.lvalues.size()))) ||
            (lvalue.kind == 4 && lvalue.a >= parsed.expressions.size()) ||
            (lvalue.kind == 5 && lvalue.a >= parsed.expressions.size()) ||
            (lvalue.kind == 6 && lvalue.a >= parsed.lvalues.size()) ||
            (lvalue.kind == 7 && (lvalue.a >= parsed.lvalues.size() || lvalue.b >= parsed.expressions.size())) ||
            (lvalue.kind == 8 && (lvalue.a >= parsed.expressions.size() || lvalue.b >= parsed.expressions.size())) ||
            (lvalue.kind == 9 && (lvalue.a >= parsed.expressions.size() ||
                                  lvalue.name_id >= parsed.strings.size()))) {
            return false;
        }
    }

    for (const auto child : parsed.children) {
        if (child >= parsed.statements.size()) {
            return false;
        }
    }

    for (const auto argument : parsed.arguments) {
        if (argument >= parsed.expressions.size()) {
            return false;
        }
    }

    for (const auto& parameter : parsed.parameters) {
        if (parameter.direction > 2 || parameter.name_id >= parsed.strings.size() || parameter.type_id >= parsed.types.size()) {
            return false;
        }
    }

    if (!validate_struct_layouts(parsed)) {
        return false;
    }

    *module = std::move(parsed);
    return true;
}

bool ValidateSection(const unsigned char* payload, uint64_t byte_length) {
    Module ignored;
    return ParseSection(payload, byte_length, &ignored);
}

} // namespace Feather::TypedIR
