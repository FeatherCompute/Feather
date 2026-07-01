#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dotnet test "$ROOT/tests/Feather.Native.Tests/Feather.Native.Tests.csproj" -v minimal
dotnet test "$ROOT/tests/Feather.Generator.Tests/Feather.Generator.Tests.csproj" -v minimal
dotnet test "$ROOT/tests/Feather.Tests/Feather.Tests.csproj" -v minimal

if [[ "${FEATHER_RUN_GPU_TESTS:-0}" == "1" ]]; then
    dotnet test "$ROOT/tests/Feather.Gpu.Tests/Feather.Gpu.Tests.csproj" -v minimal
    dotnet test "$ROOT/tests/Feather.Graphics.Tests/Feather.Graphics.Tests.csproj" -v minimal
    dotnet test "$ROOT/tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj" -v minimal
    dotnet test "$ROOT/tests/Feather.AD.Tests/Feather.AD.Tests.csproj" -v minimal
    dotnet test "$ROOT/tests/Feather.NN.Tests/Feather.NN.Tests.csproj" -v minimal
else
    echo "Skipping GPU/native integration tests. Set FEATHER_RUN_GPU_TESTS=1 to include them."
fi
