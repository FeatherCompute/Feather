# Contributing To Feather

Thanks for taking the time to improve Feather. The project is still in preview,
so contributions that make behavior clearer, better tested, or easier to run on
a fresh machine are especially valuable.

## Development Setup

```bash
git submodule update --init --recursive
./eng/build-native.sh
dotnet build Feather.slnx
```

If the managed runtime cannot find the native library, set:

```bash
export FEATHER_NATIVE_LIBRARY="$PWD/native/build/libfeather.dylib"
```

Use `libfeather.so` on Linux and `feather.dll` on Windows.

## Before Opening A Pull Request

Run the checks that match the area you changed:

```bash
./eng/build.sh
./eng/test.sh
./eng/check-markdown-links.py
```

For native or packaging changes, also run:

```bash
./eng/pack.sh
```

GPU, windowing, and graphics tests can depend on local drivers and display
services. If a test cannot run in your environment, mention the platform,
backend, and error in the pull request.

## Repository Hygiene

- Do not commit `bin/`, `obj/`, `native/build*`, `artifacts/`, crash dumps, or
  IDE workspace files.
- Do not commit Sponza scene assets to the Feather repository. Keep them in a
  local `Sponza/` directory or another external asset location.
- Do not commit native runtime binaries under `src/Feather.NativeAssets`; CI and
  `eng/pack.sh` stage those files under `artifacts/native-assets`.
- Keep public API changes intentional and update docs/tests with the change.

## Coding Style

Feather uses nullable reference types, implicit usings, central package
management, and repository-level MSBuild properties. Prefer small, focused
changes that match the surrounding style.
