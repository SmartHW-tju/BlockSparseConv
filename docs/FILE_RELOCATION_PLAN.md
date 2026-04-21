# File Relocation Plan

This document records the repository reorganization from the original flat `src/` layout to the current structure.

| Original Path | New Path | Category | Action | Reason |
| --- | --- | --- | --- | --- |
| `src/network.cpp` | `hls/src/core/network.cpp` | Active kernel core | Move | Top-level HLS kernel and schedule belong in the active HLS core path. |
| `src/pw_1x1_sparse.cpp` | `hls/src/ops/pw_1x1_sparse.cpp` | Active HLS op | Move | Core sparse pointwise operator. |
| `src/pw_1x1.cpp` | `hls/src/ops/pw_1x1.cpp` | Active HLS op | Move | Dense classifier operator. |
| `src/pw_3x3.cpp` | `hls/src/ops/pw_3x3.cpp` | Active HLS op | Move | Dense image-stem convolution. |
| `src/dw_3x3.cpp` | `hls/src/ops/dw_3x3.cpp` | Active HLS op | Move | Dense depthwise operator. |
| `src/other_layers.cpp` | `hls/src/ops/other_layers.cpp` | Active HLS helper | Move | Shared post-processing layers. |
| `src/network.hpp` | `include/network.hpp` | Active shared header | Move | Shared kernel interface should live in `include/`. |
| `src/host.cpp` | `host/src/host.cpp` | Active host | Move | Active OpenCL/XRT launcher should be separate from HLS code. |
| `src/host.hpp` | `include/host.hpp` | Active shared header | Move | Host declarations belong in active include path. |
| `src/network_tb.cpp` | `tests/csim/network_tb.cpp` | Test bench | Move | C-simulation harness. |
| `src/network_tb_0.cpp` | `tests/csim/network_tb_0.cpp` | Test bench | Move | Older C-simulation harness. |
| `src/define.hpp` | `tests/include/define.hpp` | Test-only header | Move | Used by retained test benches, not the active host/kernel path. |
| `src/top.cpp` | `experiments/host_tools/top.cpp` | Experimental host tool | Move | Auxiliary parameter-packing utility, not main runtime path. |
| `src/top.hpp` | `experiments/include/top.hpp` | Experimental header | Move | Used by the experimental host tool. |
| `src/parameter_reorder.cpp` | `archive/deprecated/parameter_reorder.cpp` | Deprecated | Move | Non-buildable due to invalid identifiers. |
| `src/suibian.cpp` | `archive/deprecated/suibian.cpp` | Deprecated | Move | Scratch/debris file, not maintainable active code. |
| `src/kernel_host.cpp` | `archive/deprecated/kernel_host.cpp` | Deprecated | Move | Unrelated GEMM-style host template. |
| `src/malloc_removed.cpp` | `archive/deprecated/malloc_removed.cpp` | Deprecated | Move | Empty placeholder. |
| `src/zcu102.cfg` | `configs/vitis/zcu102.cfg` | Build config | Move | Build config should be isolated from source code. |
| `src/img1.txt` | `data/samples/img1.txt` | Sample data | Move | Runtime/sample data should not live beside source code. |
| `src/golden.txt` | `data/samples/golden.txt` | Sample data | Move | Data belongs under `data/`; role remains uncertain. |
| `src/utils/ap_*.h` | `third_party/xilinx_ap_headers/` | Vendor header | Move | Vendored third-party headers should not live inside active source directories. |
