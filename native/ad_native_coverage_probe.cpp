#include <AD/AdjointGenerator.h>
#include <AD/GradientTape.h>
#include <IR/Builder/Builder.h>
#include <IR/Builder/BuilderContext.h>
#include <IR/Node/ArrayAccess.h>
#include <IR/Node/Call.h>
#include <IR/Node/CallInst.h>
#include <IR/Node/CompoundAssignment.h>
#include <IR/Node/LoadLocalVariable.h>
#include <IR/Node/LoadUniform.h>
#include <IR/Node/LocalVariable.h>
#include <IR/Node/Operation.h>
#include <IR/Node/Return.h>
#include <IR/Node/Store.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

class ProbeContext final : public GPU::IR::Builder::BuilderContext {
public:
	void PushTranslatedCode(std::string code) override {
		_code.push_back(std::move(code));
	}

	std::string AssignVarName() override {
		return "probe_v" + std::to_string(_nextVar++);
	}

	bool HasStructDefinition(const std::string &typeName) const override {
		return _structNames.find(typeName) != _structNames.end();
	}

	void AddStructDefinition(const std::string &typeName, const std::string &definition) override {
		_structNames[typeName] = _structDefinitions.size();
		_structDefinitions.push_back(definition);
	}

	const std::vector<std::string> &GetStructDefinitions() const override {
		return _structDefinitions;
	}

	uint32_t AllocateBindingSlot() override {
		return _nextBinding++;
	}

	void RegisterBuffer(uint32_t binding, const std::string &, const std::string &, int) override {
		_bufferBindings.push_back(binding);
	}

	std::string GetBufferDeclarations() const override {
		return {};
	}

	const std::vector<uint32_t> &GetBufferBindings() const override {
		return _bufferBindings;
	}

	void BindRuntimeBuffer(uint32_t binding, GPU::Backend::BufferHandle bufferHandle) override {
		_runtimeBuffers[binding] = bufferHandle;
	}

	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeBufferBindings() const override {
		return _runtimeBuffers;
	}

	uint32_t AllocateTextureBinding() override {
		return _nextBinding++;
	}

	void RegisterTexture(
		uint32_t binding,
		GPU::IR::Builder::PixelFormat,
		const std::string &,
		uint32_t,
		uint32_t,
		bool) override {
		_textureBindings.push_back(binding);
	}

	std::string GetTextureDeclarations() const override {
		return {};
	}

	const std::vector<uint32_t> &GetTextureBindings() const override {
		return _textureBindings;
	}

	void BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) override {
		_runtimeTextures[binding] = textureHandle;
	}

	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeTextureBindings() const override {
		return _runtimeTextures;
	}

	std::string RegisterUniform(
		const std::string &, void *, size_t, size_t,
		std::function<void(uint32_t, const std::string &, void *)>,
		std::function<void(void *, void *)>) override {
		return {};
	}

	std::string GetUniformDeclarations() const override {
		return {};
	}

	void AddCallableDeclaration(const std::string &declaration) override {
		_callableDeclarations.push_back(declaration);
	}

	void AddCallableBodyGenerator(std::function<void()> generator) override {
		_callableGenerators.push_back(std::move(generator));
	}

	void PushCallableBody() override {
		_inCallableBody = true;
	}

	void PopCallableBody() override {
		_inCallableBody = false;
	}

	std::vector<std::string> GetCallableDeclarations() const override {
		return _callableDeclarations;
	}

	std::string GenerateCallableBodies() override {
		for (const auto &generator : _callableGenerators) {
			generator();
		}
		return {};
	}

private:
	uint32_t _nextBinding = 0;
	int _nextVar = 0;
	bool _inCallableBody = false;
	std::vector<std::string> _code;
	std::vector<std::string> _structDefinitions;
	std::unordered_map<std::string, size_t> _structNames;
	std::vector<uint32_t> _bufferBindings;
	std::vector<uint32_t> _textureBindings;
	std::unordered_map<uint32_t, uint32_t> _runtimeBuffers;
	std::unordered_map<uint32_t, uint32_t> _runtimeTextures;
	std::vector<std::string> _callableDeclarations;
	std::vector<std::function<void()>> _callableGenerators;
};

std::unique_ptr<GPU::IR::Node::LoadLocalVariableNode> Local(const std::string &name) {
	return std::make_unique<GPU::IR::Node::LoadLocalVariableNode>(name);
}

std::unique_ptr<GPU::IR::Node::LoadUniformNode> Literal(const std::string &value) {
	return std::make_unique<GPU::IR::Node::LoadUniformNode>(value);
}

bool Contains(const std::string &text, const std::string &needle) {
	return text.find(needle) != std::string::npos;
}

bool ContainsLine(const std::vector<std::string> &lines, const std::string &needle) {
	for (const auto &line : lines) {
		if (Contains(line, needle)) {
			return true;
		}
	}
	return false;
}

std::string JoinLines(const std::vector<std::string> &lines) {
	std::string joined;
	for (const auto &line : lines) {
		joined += line;
		joined += '\n';
	}
	return joined;
}

int Fail(const std::string &message) {
	std::cerr << "AD native coverage probe failed: " << message << '\n';
	return 1;
}

void RecordDirectNodes(GPU::AD::GradientTape &tape) {
	using namespace GPU::IR::Node;

	tape.RegisterParameter("x", "float");
	tape.RegisterParameter("rhs", "float");

	tape.Record(CompoundAssignmentNode(CompoundAssignmentCode::AddAssign, Local("x"), Local("rhs")), true);
	tape.Record(CompoundAssignmentNode(CompoundAssignmentCode::MulAssign, Local("x"), Local("rhs")), true);
	tape.Record(LocalVariableNode("aliasOut", "float", Local("x")), true);
	tape.Record(StoreNode(Local("copyOut"), Local("x")), true);
	tape.Record(StoreNode(
		Local("arrOut"),
		std::make_unique<ArrayAccessNode>(Local("bufferLike"), Literal("0"))), true);
	tape.Record(ReturnNode(), true);
}

void RecordCallableSubTapes(GPU::AD::GradientTape &tape) {
	using namespace GPU::IR::Node;

	auto &builder = GPU::IR::Builder::Builder::Get();

	tape.PushSubTape("outer_probe");
	builder.SetInCallableBody(true);
	tape.Record(LocalVariableNode(
		"outerRet",
		"float",
		std::make_unique<OperationNode>(OperationCode::Mul, Local("p0"), Local("p0"))), true);

	tape.PushSubTape("inner_probe");
	tape.Record(LocalVariableNode(
		"innerRet",
		"float",
		std::make_unique<OperationNode>(OperationCode::Add, Local("p0"), Local("p0"))), true);
	tape.Record(ReturnNode(Local("innerRet")), true);
	tape.PopSubTape();

	tape.Record(ReturnNode(Local("outerRet")), true);
	builder.SetInCallableBody(false);
	tape.PopSubTape();

	std::vector<std::unique_ptr<Node>> arguments;
	arguments.push_back(Local("x"));
	tape.Record(StoreNode(Local("callOut"), std::make_unique<CallNode>("outer_probe", std::move(arguments))), true);
	tape.Record(StoreNode(
		Local("callOut"),
		std::make_unique<OperationNode>(
			OperationCode::Add,
			Local("callOut"),
			std::make_unique<ArrayAccessNode>(Local("probeBuffer"), Literal("0")))), true);
	tape.MarkLoss("callOut", "float");
}

void RecordControlFlow(GPU::AD::GradientTape &tape) {
	tape.BeginIfBranch("x > 0.0");
	tape.BeginElifBranch("x == 0.0");
	tape.BeginElseBranch();
	tape.EndIfChain();
}

bool ProbeBody(const GPU::AD::GradientTape &tape, const GPU::AD::GradientTape &cloned) {
	int outerIndex = -1;
	if (tape.FindSubTapeByCallableName("outer_probe", &outerIndex) == nullptr || outerIndex < 0) {
		Fail("outer_probe callable sub-tape was not found by name");
		return false;
	}

	int innerIndex = -1;
	const auto &outer = tape.SubTape(outerIndex);
	if (outer.FindSubTapeByCallableName("inner_probe", &innerIndex) == nullptr || innerIndex < 0) {
		Fail("inner_probe nested callable sub-tape was not found by name");
		return false;
	}

	if (cloned.FindSubTapeByCallableName("outer_probe") == nullptr) {
		Fail("cloned tape lost callable-name lookup metadata");
		return false;
	}

	GPU::AD::AdjointGenerator generator;
	const auto body = generator.GenerateBody(tape, true);
	const auto lines = JoinLines(body.lines);

	if (body.declarations.empty()) {
		Fail("GenerateBody produced no declarations");
		return false;
	}
	if (body.lines.empty()) {
		Fail("GenerateBody produced no adjoint body lines");
		return false;
	}
	if (body.writebacks.empty()) {
		Fail("GenerateBody produced no scalar parameter writebacks");
		return false;
	}
	if (body.bufferWritebacks.empty()) {
		Fail("GenerateBody produced no buffer parameter writebacks");
		return false;
	}
	if (!ContainsLine(body.lines, "d_x += ")) {
		Fail("generated body does not accumulate into d_x");
		return false;
	}
	if (!ContainsLine(body.lines, "d_rhs += ")) {
		Fail("generated body does not accumulate into d_rhs");
		return false;
	}
	if (!ContainsLine(body.lines, "grad_probeBuffer")) {
		Fail("generated body does not touch buffer-parameter adjoint storage");
		return false;
	}
	if (!Contains(lines, "callOut")) {
		Fail("generated body does not propagate through callable output");
		return false;
	}
	if (!Contains(lines, "if (x > 0.0)") || !Contains(lines, "else if (x == 0.0)") || !Contains(lines, " else {")) {
		Fail("generated body does not contain a structurally wrapped if/else-if/else chain");
		return false;
	}
	if (!Contains(lines, "_ca")) {
		Fail("generated body does not include remapped callable adjoint fragments");
		return false;
	}

	return true;
}

} // namespace

int main() {
	ProbeContext context;
	auto &builder = GPU::IR::Builder::Builder::Get();
	GPU::IR::Builder::Builder::ScopedBind bind(builder, context);

	GPU::AD::GradientTape tape;
	GPU::IR::Builder::Builder::ScopedGradientTape tapeScope(builder, &tape);

	tape.RegisterBufferParameter("probeBuffer", "float", 1);
	tape.RegisterBufferParameter("probeBuffer", "float", 4);
	tape.RegisterBufferAdjointStorage("probeAux", "float", 1);
	tape.RegisterBufferAdjointStorage("probeAux", "float", 4);

	RecordDirectNodes(tape);
	RecordCallableSubTapes(tape);
	RecordControlFlow(tape);

	GPU::AD::GradientTape cloned;
	cloned.CloneSubTapesFrom(tape);

	if (tape.Size() == 0) {
		return Fail("forward tape is empty");
	}
	if (cloned.SubTapeCount() == 0) {
		return Fail("cloned tape has no callable sub-tapes");
	}

	return ProbeBody(tape, cloned) ? 0 : 1;
}
