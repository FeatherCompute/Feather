#!/usr/bin/env python3
"""Run the Feather NN device-path validation gate."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(command: list[str], root: Path) -> bool:
    print(f"\n==> {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    return result.returncode == 0


def source_gate(root: Path) -> bool:
    checks = [
        (
            "Linear.Forward",
            root / "src/Feather/NN/Modules.cs",
            r"public Tensor<float> Forward\(Tensor<float> input\)(.*?)^\s*}\n\n    /// <inheritdoc />",
            ["ToArray(", "Upload("],
        ),
        (
            "Optimizer.Step implementations",
            root / "src/Feather/NN/Modules.cs",
            r"public abstract class Optimizer.*?internal static class FloatParameterFactory",
            ["Value.Buffer.ToArray(", "Gradient.Buffer.ToArray(", "parameter.Value.Buffer.Upload("],
        ),
        (
            "Activation.Forward",
            root / "src/Feather/NN/ActivationsAndLosses.cs",
            r"public Tensor<float> Forward\(Tensor<float> input\)(.*?)^\s*}",
            ["ToArray(", "Upload("],
        ),
        (
            "TensorOps",
            root / "src/Feather/NN/ActivationsAndLosses.cs",
            r"public static class TensorOps(.*?)^\s*/// <summary>\n/// Device-backed loss functions",
            ["ToArray(", "Upload(", ".Read("],
        ),
        (
            "Device tensor losses",
            root / "src/Feather/NN/ActivationsAndLosses.cs",
            r"public static Tensor<float> MeanSquaredErrorTensor.*?public static float MeanSquaredError",
            ["ToArray(", "Upload(", ".Read("],
        ),
        (
            "Device cross entropy tensor losses",
            root / "src/Feather/NN/ActivationsAndLosses.cs",
            r"public static Tensor<float> CrossEntropyTensor.*?public static float CrossEntropy\(Tensor<float> probabilities, Tensor<int> labels\)",
            ["ToArray(", "Upload(", ".Read("],
        ),
        (
            "Device logits cross entropy tensor loss",
            root / "src/Feather/NN/ActivationsAndLosses.cs",
            r"public static Tensor<float> CrossEntropyFromLogitsTensor.*?public static float CrossEntropyFromLogits\(Tensor<float> logits, Tensor<int> labels\)",
            ["ToArray(", "Upload(", ".Read("],
        ),
        (
            "NN device ops",
            root / "src/Feather/NN/DeviceOps.cs",
            r"\A(.*)\Z",
            ["ToArray(", ".Read(", ".Upload("],
        ),
        (
            "AD device handoff",
            root / "src/Feather/AD/AD.cs",
            r"public void CopyGradientToBuffer.*?^\s*}\n\n    /// <inheritdoc />",
            ["fe_kernel_read_ad_gradient", "ToArray(", "Upload("],
        ),
    ]

    passed = True
    for name, path, pattern, forbidden in checks:
        text = path.read_text()
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if not match:
            print(f"source gate failed: could not locate {name} in {path}")
            passed = False
            continue

        body = match.group(1) if match.lastindex else match.group(0)
        found = [token for token in forbidden if token in body]
        if found:
            print(f"source gate failed: {name} contains forbidden production readback/update token(s): {', '.join(found)}")
            passed = False

    device_ops = (root / "src/Feather/NN/DeviceOps.cs").read_text()
    if "NnSumToScalarKernel" in device_ops:
        print("source gate failed: NN reductions must not use the old single-invocation NnSumToScalarKernel.")
        passed = False
    if "ThreadGroupSize(1, 1, 1)" in device_ops and re.search(r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*count\.Value", device_ops):
        print("source gate failed: NN reductions contain a single-thread full-tensor count.Value loop.")
        passed = False
    if "NnPartialSumKernel" not in device_ops or "Reduce.PartialSum" not in device_ops:
        print("source gate failed: NN reductions must use the scalable partial-sum GPU path.")
        passed = False

    gpu_kernel = (root / "src/Feather/Kernels/GpuKernel.cs").read_text()
    native_structs = (root / "src/Feather.Native/NativeStructs.cs").read_text()
    native_api = (root / "native/feather_c_api.cpp").read_text()
    native_lowerer = (root / "native/feather_typed_ir_lowerer.cpp").read_text()
    bounds_requirements = [
        ("managed kernel create descriptor", "descriptor.BoundsCheck" in gpu_kernel),
        ("managed logical dispatch forwarding", "size.X" in gpu_kernel and "logical_x" in (root / "src/Feather.Native/NativeMethods.cs").read_text()),
        ("native create descriptor bounds flag", "BoundsCheck" in native_structs and "bounds_check" in (root / "native/feather_c_api.h").read_text()),
        ("native logical dispatch storage", "logical_x" in native_api and "logical_y" in native_api and "logical_z" in native_api),
        ("typed lowerer guard", "EmitBoundsCheckGuard" in native_lowerer and "builder_.Compare(GPU::IR::CompareOp::GreaterEqual" in native_lowerer),
        ("hidden logical dispatch constants", "__feather_dispatch_size_" in native_lowerer),
    ]
    for name, ok in bounds_requirements:
        if not ok:
            print(f"source gate failed: bounds checking is not wired through {name}.")
            passed = False

    modules = (root / "src/Feather/NN/Modules.cs").read_text()
    if "[EditorBrowsable(EditorBrowsableState.Never)]" not in modules or "StepFromDebugGradients" not in modules:
        print("source gate failed: Optimizer.Step(GradientSet) must be hidden and exposed as an explicit debug/interop path.")
        passed = False
    if "support Parameter<float> only" not in modules:
        print("source gate failed: NN optimizers must reject unsupported non-float parameters clearly.")
        passed = False
    if "FindGradientMatches" not in modules or "Multiple native AD gradients matched" not in modules:
        print("source gate failed: device AD handoff must validate native gradient aliases before copying.")
        passed = False

    training_tests = (root / "tests/Feather.Integration.Tests/NNTrainingIntegrationTests.cs").read_text()
    for token in ("GradientSet", "StepFromDebugGradients", ".Gradients.Get<", ".Gradients.GetArray<", ".Gradients.TryGet"):
        if token in training_tests:
            print(f"source gate failed: normal NN training tests must not use managed GradientSet readback token {token!r}.")
            passed = False
    for token in (
        "SequentialReluMlpTrainingStepWithAdamUsesModuleOwnedParameters",
        "ModuleBackedBinaryClassifierWithAdamWDecreasesCrossEntropy",
        "OptimizerStepAdKernelReportsMissingMismatchedAmbiguousAndUnsupportedNativeGradients",
    ):
        if token not in training_tests:
            print(f"source gate failed: missing required NN training/guardrail test {token}.")
            passed = False

    nn_tests = (root / "tests/Feather.NN.Tests/NNSurfaceTests.cs").read_text()
    for token in (
        "SgdMatchesCpuReferenceAcrossIndustrialEdgeCases",
        "SgdMomentumMatchesCpuReferenceAcrossIndustrialEdgeCases",
        "RmsPropMatchesCpuReferenceAcrossIndustrialEdgeCases",
        "AdamMatchesCpuReferenceAcrossIndustrialEdgeCases",
        "AdamWMatchesCpuReferenceAcrossIndustrialEdgeCases",
        "OptimizerRejectsUnsupportedNonFloatParameters",
    ):
        if token not in nn_tests:
            print(f"source gate failed: missing required optimizer industrial test {token}.")
            passed = False

    sequence_models = (root / "src/Feather/NN/SequenceModels.cs").read_text()
    if re.search(r"public\s+.*PredictNext\(", sequence_models) or re.search(r"public\s+.*Forward\(", sequence_models):
        print("source gate failed: SequenceModels public inference methods must be explicitly named as host helpers.")
        passed = False
    for token in ("ForwardHost", "RunHost", "PredictNextHost", "PredictHost"):
        if token not in sequence_models:
            print(f"source gate failed: missing explicit host inference helper {token}.")
            passed = False

    nn_source = "\n".join(path.read_text() for path in (root / "src/Feather/NN").glob("*.cs"))
    if re.search(r"public\s+.*\b(ADKernel|Scratch|Tokens|Features|Labels)\b", nn_source):
        print("source gate failed: trainer internals must not be public NN API.")
        passed = False
    for raw_public_buffer in (
        "public GpuBuffer<float> Loss",
        "public GpuBuffer<int> Tokens",
        "public GpuBuffer<float> Features",
        "public GpuBuffer<float> Labels",
        "public GpuADKernel",
    ):
        if raw_public_buffer in nn_source:
            print(f"source gate failed: raw trainer internals exposed via {raw_public_buffer}.")
            passed = False

    samples_tests_and_nn = "\n".join(
        path.read_text()
        for folder in ("samples", "tests", "src/Feather/NN")
        for path in (root / folder).rglob("*.cs")
    )
    if re.search(r"trainer\.ADKernel|\.(Scratch|Tokens|Features|Labels)\b", samples_tests_and_nn):
        print("source gate failed: samples/tests must use trainer diagnostics instead of raw internals.")
        passed = False
    for token in ("LastDispatchPath", "GradientsMaterialized", "LastLoss"):
        if token not in nn_source:
            print(f"source gate failed: trainer diagnostic {token} is missing.")
            passed = False

    for token in (
        "SequentialForwardDisposesIntermediateTensorsAcrossRepeatedCalls",
        "SequentialForwardDisposesOwnedIntermediateWhenLaterModuleThrows",
    ):
        if token not in nn_tests:
            print(f"source gate failed: missing Sequential lifetime regression test {token}.")
            passed = False

    for sample in ("AdGptDemo/Program.cs", "AdGptPoetDemo/Program.cs"):
        text = (root / "samples" / sample).read_text()
        for token in ("model-only eval loss delta", "model-only generated", "prior-assisted generated", "Expected model-only eval loss to decrease"):
            if token not in text:
                print(f"source gate failed: {sample} must report model-only loss/generation separately from prior-assisted output.")
                passed = False

    return passed


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="rebuild native library before running tests")
    args = parser.parse_args(argv)
    root = repo_root()

    ok = source_gate(root)
    if args.clean:
        ok = run(["cmake", "--build", "native/build"], root) and ok

    steps = [
        ["dotnet", "test", "tests/Feather.NN.Tests/Feather.NN.Tests.csproj", "--no-restore"],
        [
            "dotnet",
            "test",
            "tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj",
            "--no-restore",
            "--filter",
            "NN|Training|AutoDiff|AD|Bounds",
        ],
    ]

    for step in steps:
        ok = run(step, root) and ok

    if not ok:
        print("\nNN industrial gate failed.", file=sys.stderr)
        return 1

    print("\nNN industrial gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
