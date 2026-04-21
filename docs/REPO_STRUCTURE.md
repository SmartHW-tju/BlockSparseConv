# Repository Structure Guide

## Purpose

This document explains the reorganized repository layout and the reasoning behind each top-level directory.

The repository was restructured around four priorities:

- preserve the active paper-reproduction path
- separate stable implementation from historical exploration
- make synthesis-sensitive code easy to find
- reduce ambiguity between source, tests, runtime data, and deprecated material

## Top-Level Directories

### `hls/`

Active hardware implementation.

- `hls/src/core/`
  Contains the top-level kernel and end-to-end scheduler.
- `hls/src/ops/`
  Contains reusable operator implementations: sparse pointwise, dense depthwise, dense stem, and helper layers.

### `host/`

Active host/runtime application.

- `host/src/host.cpp`
  Loads exported tensors, builds OpenCL buffers, launches the kernel, and validates output.

### `include/`

Headers shared by the active kernel and host paths.

- `network.hpp`
  Active HLS/kernel interface, shared types, and function declarations.
- `host.hpp`
  Host-side aligned allocator and large host buffer declarations.

### `tests/`

Retained validation and C-simulation code.

- `tests/csim/`
  C-simulation-style test benches that directly invoke the kernel path.
- `tests/include/`
  Test-only buffer declarations currently used by the legacy test benches.

### `experiments/`

Historical but still potentially informative code.

- `experiments/kernel_variants/`
  Earlier or dated kernel snapshots such as `network_0.cpp`, `network_new.cpp`, and `h0903_network_new.cpp`.
- `experiments/host_tools/`
  Auxiliary host-side utilities such as `top.cpp`.
- `experiments/include/`
  Headers used only by the experimental host tools.

### `archive/`

Files that should not sit next to active code.

- `archive/deprecated/`
  Broken, placeholder, or unrelated files preserved only for traceability.

### `configs/`

Build and synthesis configuration fragments.

- `configs/vitis/zcu102.cfg`
  The checked-in Vitis platform configuration.

### `data/`

Runtime tensors and checked-in samples.

- `data/runtime/`
  Intended location for the full exported tensor/metadata set needed by the host or test benches.
- `data/samples/`
  Small checked-in example files from the current snapshot.

### `docs/`

Project documentation.

- `README.md` at repo root is the user entry point.
- `docs/REPO_STRUCTURE.md` explains layout and migration.
- `docs/MODULE_GUIDE.md` explains architecture and module roles.
- `docs/FILE_RELOCATION_PLAN.md` records old-to-new file moves.
- `docs/TODO.md` tracks remaining cleanup and reproducibility work.

### `third_party/`

Vendored external dependencies.

- `third_party/xilinx_ap_headers/`
  Local copies of Xilinx arbitrary-precision headers that were previously mixed into source code.

## Organization Principles

- Active implementation lives in `hls/`, `host/`, and `include/`.
- Validation lives in `tests/`.
- Historical snapshots live in `experiments/`.
- Deprecated or broken files live in `archive/`.
- Runtime artifacts and exported tensors live under `data/`, not beside source files.
- Configuration lives in `configs/`.
- Vendor headers live in `third_party/`.

## Migration Notes From The Old Layout

The original repository placed almost everything under a single `src/` directory. That mixed:

- active HLS kernel code
- active host code
- test benches
- historical kernel revisions
- data samples
- build config
- vendored headers
- deprecated scratch files

The new layout resolves that by moving files into role-specific locations while preserving filenames for the active code path wherever possible.

## File Category Rules

- New active kernel code should go to `hls/src/core/` or `hls/src/ops/`.
- New active shared headers should go to `include/`.
- New test benches or hostless validation code should go to `tests/csim/`.
- New scripts for build or data preparation should go to a future `scripts/` directory rather than `experiments/`.
- Generated reports, bitstreams, and logs should go to `artifacts/` or another ignored output directory, not into source folders.

## Remaining Structural Gaps

- A dedicated `scripts/` directory does not yet exist because no standalone scripts were present in the original snapshot.
- A dedicated `artifacts/` directory is not populated yet, but `.gitignore` now reserves that pattern for generated outputs.
- Test benches still depend on large external data dumps and have not yet been refactored into self-contained unit tests.
