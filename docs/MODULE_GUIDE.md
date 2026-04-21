# Module Guide

## Overview

This guide maps the active implementation files to the accelerator architecture and describes the current dataflow at a repository-maintenance level.

The design is best understood as:

1. host-side tensor loading and AXI packing
2. top-level HLS kernel orchestration
3. sparse pointwise and dense depthwise compute primitives
4. classifier head and golden-output comparison

## Primary File Mapping

| File | Role | Notes |
| --- | --- | --- |
| `include/network.hpp` | Active shared interface | Declares fixed-point types, operator prototypes, bottleneck helpers, and the `extern "C"` kernel entry. |
| `hls/src/core/network.cpp` | Top-level HLS kernel | Implements AXI unpacking/packing helpers, local scratch buffers, bottleneck scheduling, residual spill/reload, and the `network(...)` entry point. |
| `hls/src/ops/pw_1x1_sparse.cpp` | Sparse convolution | Implements block-sparse `1x1` pointwise convolution for square and general feature maps. |
| `hls/src/ops/dw_3x3.cpp` | Depthwise convolution | Implements dense depthwise `3x3` kernels for stride 1 and stride 2. |
| `hls/src/ops/pw_3x3.cpp` | Stem convolution | Implements the dense image-stem `3x3` convolution and its split helper forms. |
| `hls/src/ops/pw_1x1.cpp` | Dense classifier | Implements the final dense `1x1` classifier after global average pooling. |
| `hls/src/ops/other_layers.cpp` | Helper layers | Implements batch norm, ReLU6, and global average pooling. |
| `include/host.hpp` | Host shared declarations | Declares aligned allocators and the large host-side data buffers used by the OpenCL launcher. |
| `host/src/host.cpp` | Active runtime path | Loads text-format tensors, packs them into AXI words, launches the FPGA kernel, and checks results against a golden file. |

## Architecture Notes

### 1. Host Data Preparation

`host/src/host.cpp` reads exported text files for:

- input image
- sparse `1x1` weights
- dense `3x3` weights
- batch-normalization parameters
- classifier shards
- sparse block indices `block_c`
- sparse prefix arrays `block_col_r`
- golden output logits

The host then packs most tensors into `ap_uint<512>` AXI words before launching `network(...)`.

### 2. Kernel Entry And Memory Interfaces

`network(...)` in `hls/src/core/network.cpp` is the synthesis-critical entry point.

Its interface exposes:

- packed input image
- sparse weights and sparse metadata
- dense depthwise weights
- classifier weight shards and bias
- per-layer sparse prefix arrays
- spill/reload feature-map buffers
- output buffer

The kernel uses many `#pragma HLS INTERFACE m_axi` declarations, so argument order and port grouping are part of the effective ABI.

### 3. Sparse Pointwise Dataflow

`pw_1x1_sparse` uses:

- `weight`
  Packed `1x4` output-channel blocks.
- `block_c`
  Flattened input-channel indices for nonzero sparse blocks.
- `block_col_r`
  Prefix pointers for output-channel groups.

Operationally:

1. input feature maps are rearranged into an `img2col`-style layout
2. each output-channel group reads its nonzero block range from `block_col_r`
3. the kernel accumulates only referenced input channels from `block_c`
4. four output channels are produced per sparse block

This is the core block-sparse primitive in the repository.

### 4. Dense Depthwise Dataflow

`dw_3x3_s1` and `dw_3x3_s2` in `hls/src/ops/dw_3x3.cpp` implement dense depthwise stages.

- `dw_3x3_s1`
  Uses explicit boundary handling to avoid constructing a fully padded tensor.
- `dw_3x3_s2`
  Uses a padded-buffer style better suited to the stride-2 access pattern.

These operators are paired with batch norm and ReLU6 in the bottleneck helpers.

### 5. Bottleneck Scheduling

`hls/src/core/network.cpp` contains:

- AXI unpack/pack helpers
- `bottleneck`
- `bottleneck_1`
- `bottleneck_2`
- stripe/partition helpers for the early large feature-map stages

The schedule is a MobileNetV2-like sequence of:

- stem convolution
- depthwise + sparse projection
- repeated inverted residual blocks
- final sparse expansion to `1280`
- global average pooling
- dense classifier head

### 6. Spill Buffers And Capacity Management

The kernel does not keep the entire network live on-chip.

Instead it uses:

- local scratch buffers for currently active feature maps
- explicit DDR spill/reload for selected residual branches
- dedicated intermediate buffers such as `feature_map_raw`, `fm_16_1_raw`, and `feature_map_2_raw`

This is important for understanding both performance and repository organization: the memory-management helpers in `network.cpp` are part of the architecture, not just utility code.

## Historical Modules

These are not the default path but remain useful for reverse engineering:

- `experiments/kernel_variants/network_0.cpp`
- `experiments/kernel_variants/network_1.cpp`
- `experiments/kernel_variants/network_new.cpp`
- `experiments/kernel_variants/network_new_0911.cpp`
- `experiments/kernel_variants/h0901_network_new.cpp`
- `experiments/kernel_variants/h0903_network_new.cpp`
- `experiments/kernel_variants/orig_h0901_network_new.cpp`

They appear to represent older scheduling experiments, interface revisions, and dated optimization attempts.

## Test And Validation Files

- `tests/csim/network_tb.cpp`
- `tests/csim/network_tb_0.cpp`

These test benches directly call the kernel path and load the same style of external tensor dump files as the active host.

They are best treated as historical validation harnesses rather than clean unit tests.

## Uncertain Or Non-Primary Files

- `data/samples/golden.txt`
  Present in the snapshot but not used by the active host path.
- `experiments/host_tools/top.cpp`
  Useful as a host-side packing reference, but not the primary launcher.
- `archive/deprecated/parameter_reorder.cpp`
  Intended as a packing helper, but invalid C++ identifiers make it non-buildable as checked in.

## Practical Interface Summary

If you are extending the active design, start here:

- types and ABI: `include/network.hpp`
- top-level schedule: `hls/src/core/network.cpp`
- sparse operator: `hls/src/ops/pw_1x1_sparse.cpp`
- host packing/launch order: `host/src/host.cpp`

Those four locations define most of the real coupling in the current repository.
