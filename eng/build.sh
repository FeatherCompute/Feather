#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/eng/build-native.sh"
"$ROOT/eng/stage-native-assets.sh"
dotnet build "$ROOT/Feather.slnx" -v minimal
