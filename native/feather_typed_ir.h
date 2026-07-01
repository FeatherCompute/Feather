#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace Feather::TypedIR {

inline constexpr uint32_t NoIndex = UINT32_MAX;

struct Range {
    uint32_t offset = 0;
    uint32_t count = 0;
};

struct Function {
    uint8_t kind = 0;
    uint32_t name_id = NoIndex;
    uint32_t mangled_name_id = NoIndex;
    uint32_t return_type_id = NoIndex;
    uint32_t first_parameter = NoIndex;
    uint32_t parameter_count = 0;
    uint32_t body_statement_index = NoIndex;
};

struct Type {
    uint8_t kind = 0;
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t d = 0;
};

struct StructRecord {
    uint32_t name_id = NoIndex;
    uint32_t fully_qualified_name_id = NoIndex;
    uint32_t first_field = NoIndex;
    uint32_t field_count = 0;
    uint32_t size_in_bytes = 0;
    uint32_t alignment = 0;
};

struct StructField {
    uint32_t name_id = NoIndex;
    uint32_t type_id = NoIndex;
    uint32_t offset = 0;
    uint32_t size_in_bytes = 0;
    uint32_t flags = 0;
};

struct Statement {
    uint8_t kind = 0;
    uint32_t a = NoIndex;
    uint32_t b = NoIndex;
    uint32_t c = NoIndex;
    uint32_t op = 0;
    uint32_t name_id = NoIndex;
    uint32_t first_child = NoIndex;
    uint32_t child_count = 0;
};

struct Expression {
    uint8_t kind = 0;
    uint32_t type_id = NoIndex;
    uint32_t a = NoIndex;
    uint32_t b = NoIndex;
    uint32_t c = NoIndex;
    uint32_t name_id = NoIndex;
    uint32_t op = 0;
    uint32_t first_argument = NoIndex;
    uint32_t argument_count = 0;
};

struct LValue {
    uint8_t kind = 0;
    uint32_t type_id = NoIndex;
    uint32_t a = NoIndex;
    uint32_t b = NoIndex;
    uint32_t c = NoIndex;
    uint32_t name_id = NoIndex;
};

struct Parameter {
    uint8_t direction = 0;
    uint32_t name_id = NoIndex;
    uint32_t type_id = NoIndex;
};

struct Callable {
    std::string name;
    uint32_t function_index = NoIndex;
};

struct Module {
    uint32_t entry_function = NoIndex;
    std::vector<Function> functions;
    std::vector<Type> types;
    std::vector<StructRecord> structs;
    std::vector<StructField> struct_fields;
    std::vector<Statement> statements;
    std::vector<Expression> expressions;
    std::vector<LValue> lvalues;
    std::vector<uint32_t> children;
    std::vector<uint32_t> arguments;
    std::vector<Parameter> parameters;
    std::vector<std::string> strings;
    std::unordered_map<std::string, Callable> callables;
};

bool ParseSection(const unsigned char* payload, uint64_t byte_length, Module* module);
bool ValidateSection(const unsigned char* payload, uint64_t byte_length);

} // namespace Feather::TypedIR
