# BlockSparseConv

FPGA/HLS implementation repository for the paper:

`An Efficient Hardware Accelerator for Block Sparse Convolutional Neural Networks on FPGA`

This repository contains a Xilinx Vitis/Vivado HLS implementation of a block-sparse CNN accelerator, an OpenCL/XRT host application, historical kernel variants kept for research traceability, and lightweight documentation for reproducing and extending the design.

## Relation To The Paper

The active implementation matches the paper's accelerator theme:

- block-sparse `1x1` pointwise convolutions
- dense depthwise `3x3` convolutions
- a MobileNetV2-like bottleneck schedule
- fixed-point HLS kernels using `ap_fixed` and `ap_uint<512>`
- external DDR traffic for packed parameters, feature-map spill buffers, and sparse metadata

The repository does not contain the full offline training/pruning/export pipeline. The hardware path is present; some parameter-export provenance remains external to this checkout.

## Current Primary Code Path

The intended main path for reproduction is:

- kernel interface and scheduler: `hls/src/core/network.cpp`
- reusable compute operators: `hls/src/ops/`
- shared kernel declarations/types: `include/network.hpp`
- OpenCL/XRT host launcher: `host/src/host.cpp`
- Vitis platform config: `configs/vitis/zcu102.cfg`

Historical snapshots remain available under `experiments/` and deprecated scratch files are quarantined under `archive/`.

## Repository Structure

```text
.
├── archive/
│   └── deprecated/            # broken or clearly deprecated scratch files
├── configs/
│   └── vitis/                 # platform and build configuration snippets
├── data/
│   ├── runtime/               # expected location for full runtime tensor dumps
│   └── samples/               # small checked-in sample files from this snapshot
├── docs/
│   ├── FILE_RELOCATION_PLAN.md
│   ├── MODULE_GUIDE.md
│   ├── REPO_STRUCTURE.md
├── experiments/
│   ├── host_tools/            # auxiliary or older host-side utilities
│   ├── include/               # headers used only by experimental tools
implementations
├── hls/
│   └── src/
│       ├── core/              # top-level kernel and schedule
│       └── ops/               # reusable compute primitives
├── host/
│   └── src/                   # OpenCL/XRT host application
├── include/                   # active shared headers
├── tests/
│   ├── csim/                  # C-simulation style test benches
│   └── include/               # test-only buffer declarations
└── third_party/
    └── xilinx_ap_headers/     # vendored arbitrary-precision headers
```

## Main Modules

- `network(...)` in `hls/src/core/network.cpp`
  Loads packed parameters from AXI, schedules the end-to-end network, manages residual spill/reload, and writes classifier output.
- `pw_1x1_sparse` in `hls/src/ops/pw_1x1_sparse.cpp`
  Implements the block-sparse `1x1` convolution using `block_c` and `block_col_r` metadata.
- `dw_3x3_s1` / `dw_3x3_s2` in `hls/src/ops/dw_3x3.cpp`
  Implement dense depthwise `3x3` layers for stride 1 and stride 2.
- `pw_3x3` in `hls/src/ops/pw_3x3.cpp`
  Implements the dense image stem convolution.
- `batchnorm2d`, `relu6`, `globalaveragepooling` in `hls/src/ops/other_layers.cpp`
  Implement the post-processing helpers used between convolution stages.
- `host/src/host.cpp`
  Loads text tensors, repacks them into `512`-bit AXI words, launches the FPGA kernel, and compares outputs with a golden reference.

## Setup Requirements

- Xilinx Vitis/Vivado HLS toolchain with OpenCL/XRT support
- C++ compiler compatible with the Xilinx host flow
- Xilinx runtime libraries (`CL/cl2.hpp`, `cl_ext_xilinx.h`, platform libraries)
- A complete runtime parameter export set

The repository vendors `ap_fixed` and `ap_int` headers under `third_party/xilinx_ap_headers/`, but production builds are still expected to use the standard Xilinx toolchain environment.

## Build / Synthesis

Example kernel compile flow after the reorganization:

```bash
v++ -c \
  --config configs/vitis/zcu102.cfg \
  -Iinclude \
  -Ithird_party/xilinx_ap_headers \
  -k network \
  hls/src/core/network.cpp \
  hls/src/ops/pw_1x1_sparse.cpp \
  hls/src/ops/pw_1x1.cpp \
  hls/src/ops/pw_3x3.cpp \
  hls/src/ops/dw_3x3.cpp \
  hls/src/ops/other_layers.cpp \
  -o artifacts/network.xo
```

Link:

```bash
v++ -l \
  --config configs/vitis/zcu102.cfg \
  artifacts/network.xo \
  -o artifacts/network.xclbin
```

Host compilation is toolchain-specific because it depends on the installed XRT/OpenCL environment.

## Run / Test

The active host now searches for runtime text tensors in this order:

1. `BSC_DATA_DIR`
2. `data/runtime/`
3. `data/samples/`
4. current working directory

Example:

```bash
export BSC_DATA_DIR=/path/to/exported-paper-runtime-data
./host artifacts/network.xclbin
```

Available checked-in sample data is incomplete. Only `data/samples/img1.txt` and `data/samples/golden.txt` are present in this snapshot, so full end-to-end runtime execution is not reproducible from the repository alone.

Test benches are retained under `tests/csim/`, but they still depend on the same external tensor dump set as the host flow.

## Reproduction Workflow

1. Prepare the complete exported tensors and sparse metadata from the original training/export pipeline.
2. Place them in `data/runtime/` or point `BSC_DATA_DIR` to that directory.
3. Build the kernel with Vitis using the files under `hls/src/`.
4. Build the host against XRT/OpenCL using `host/src/host.cpp` and `include/host.hpp`.
5. Run the host with the generated `network.xclbin`.
6. Compare the predicted logits/class index against the golden data file.

## Notes And Limitations

- The active ABI between host and kernel is large and ordering-sensitive.
- Many weight and metadata offsets are hard-coded in the current implementation.
- Historical kernel variants in `experiments/kernel_variants/` may contain useful ideas, but they are not the default reproduction path.
- `archive/deprecated/` contains files that are broken, incomplete, or unrelated to the active accelerator flow.
- The sparse tensor export process is not fully documented in the current repository snapshot.

See:

- `docs/REPO_STRUCTURE.md`
- `docs/MODULE_GUIDE.md`
- `docs/FILE_RELOCATION_PLAN.md`

## Citation

If you find this repository useful in your research, please cite the corresponding paper:

```bibtex
@article{10186275,
  author={Yin, Xiaodi and Wu, Zhipeng and Li, Dejian and Shen, Chongfei and Liu, Yu},
  journal={IEEE Embedded Systems Letters}, 
  title={An Efficient Hardware Accelerator for Block Sparse Convolutional Neural Networks on FPGA}, 
  year={2024},
  volume={16},
  number={2},
  pages={158-161},
  doi={10.1109/LES.2023.3296507}
  }

