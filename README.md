# A journey to ~2.84 TFLOPs on my M2 MacBook

This is a diary of my journey to write a fast single-floating point (FP32) matrix multiplication (SGEMM) kernel on my Apple M2 laptop. Since I don't have an NVIDIA card lying around, I'm using Apple's **Metal** API instead of CUDA. The core ideas are the same: making GEMM as fast as possible. 

The theoretical FP32 peak of a 8-core M2 is **~2.84 TFLOPs** (or 2840 GFLOPS) [^1] [^2]. Can we even get close? Let's find out.

This repo is for anyone who wants to learn GPU optimization but don't have NVIDA GPU like me. The code is (a little bit) clean, and built with modern C++ so we don't leak memory all over the place (and shoot our foot in place).

## How fast are we so far?

| Kernel | Best performance (GFLOPS) | Percent of peak performance |
|--------|---------------------------|---|
| `naive` | $\approx 178$ | $\approx 6.26\%$ |
| `tile_16` | $\approx 269$ | $\approx 9.12\%$ |
| `tile_32` | $\approx 195$ | $\approx 6.86\%$ |
| `tile_threads` | $\approx 359$ | $\approx 12.6\%$ |
| `tile_simdgroup` | $\approx 421$ | $\approx 17\%$ |

## Get it running

>[!IMPORTANT]
>You'll need **a Mac** with an **M-series chip**.

### 1. Installation

If you don't have [Homebrew](https://brew.sh/), get it. Then:
```bash
brew install cmake make llvm@20
```

>[!NOTE]
>We need `llvm`'s clang because Apple's default one can be a bit... quirky. Note that `llvm` newest version (21) cannot work on our code properly so we choose version 20.

### 2. Build the thing

Pop open a terminal and run these:
```bash
# Make a home for the benchmark results
mkdir -p outputs

# Let CMake do its magic. This points to the new clang we just installed.
cmake -S . -B build -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=$(brew --prefix llvm@20)/bin/clang \
      -DCMAKE_CXX_COMPILER=$(brew --prefix llvm@20)/bin/clang++

# Fire the lasers!
cmake --build build
```
If that all worked, you'll have a shiny new executable at `build/bin/gemm`.

## Run the benchmark

```bash
# It's easy, just tell it which kernel to run
./build/bin/gemm naive
```

The program will spit out performance numbers to your console and also save a detailed `.csv` file in the `outputs` folder. This is the good stuff you can use to make pretty graphs in Python.

**Available kernels:**
- `naive`: The humble beginning. 
- `tile_16` and `tile_32`: Tiling kernel with different tile sizes.
- `tile_threads`: Tiling kernel with more work on threads.
- `tile_simdgroup`: Tiling kernel with `simdgroup` (metal intrinsics).

## The Grand Plan (aka The Optimization Checklist)

This project is structured so you can follow the optimization journey step-by-step. Each new kernel will be a separate file, building on the lessons of the last.

- [x] **Chapter 0: The Setup** Build a solid, memory-safe C++ framework with a real benchmark harness. No segfaults allowed.
- [x] **Chapter 1: The Tiling** The first real optimization. Use that sweet, sweet shared memory or SMEM (`__threadgroup` in Metal, `__shared__` in CUDA) to stop hitting DRAM so much. This is where we should see the first big performance jump.
- [x] **Chapter 2: More Work, Less Laziness (Register Tiling)** Make each thread compute a small 2x2 or 4x4 block of the output matrix. This increases register reuse and hides instruction latency.
- [x] **Chapter 3: Embracing the Hardware (SIMD-group Matrix Primitives)** This is the game-changer. We will stop using scalar math and switch to the hardware's native matrix multiplication capabilities.
- [ ] **Chapter 4: Hiding Latency (Software Pipelining)** Overlap memory fetching with computation using `async_copy` and double-buffering in `threadgroup` memory.
- [ ] **Chapter 5: Adaptive Tiling (Specialization & Tuning)** Move from runtime parameters to compile-time constants. Implement heuristics to choose the best tile size and configuration for the target GPU and problem size.

## Performance Analysis

### Chapter 1: The tiling

#### Why we need tiling ?

> [!CAUTION]
> TODO

#### Why is `tile_16` Faster Than `tile_32`?

This result is counter-intuitive at first. A larger tile size like 32x32 should mean more data reuse within the fast `threadgroup` memory (or *shared memory*), which is usually good for performance. However, it's slower. Why?

The answer is **Occupancy**.

1.  **What is Occupancy?** Occupancy is the ratio of active threadgroups (or warps) to the maximum number of threadgroups that can run on a single GPU compute unit (CU) (or an SM in CUDA device). High occupancy is critical for hiding memory latency. When one group of threads is stalled waiting for data to arrive from the slow device memory (DRAM), the GPU scheduler can switch to another *resident* group and keep the compute units busy.

2.  **Resource Limits:** A CU has a fixed amount of resources, including registers and, most importantly for this case, `threadgroup` memory.
- `tile_16` kernel:
      - Threadgroup size: 16x16 = 256 threads.
      - `threadgroup` memory used: `(16*16 + 16*16) * 4 bytes = 2048 bytes`.
- `tile_32` kernel:
      - Threadgroup size: 32x32 = 1024 threads.
      - `threadgroup` memory used: `(32*32 + 32*32) * 4 bytes = 8192 bytes`.

3.  **The Bottleneck:** The M2 GPU's CUs have a limited amount of `threadgroup` memory (32 KB) [^2]. The `tile_32` kernel's 8KB memory footprint is significant. If a single threadgroup consumes too large a chunk of the CU's available memory, the scheduler cannot fit as many *concurrent* threadgroups onto that CU.

With `tile_32`, you might only be able to fit one or two threadgroups per CU, leading to low occupancy. If those few groups stall on a memory read, there are no other resident groups to switch to, and the expensive ALU units sit idle.

The `tile_16` kernel, with its much smaller 2KB footprint, allows many more threadgroups to be resident on the CU simultaneously. This gives the scheduler a large pool of work to choose from, effectively hiding memory latency and keeping the hardware busy.horrors.

### Chapter 2: More work on threads

> [!CAUTION]
> TODO

### Chapter 3: SIMD Tilegroup

> [!CAUTION]
> TODO

## Resources

- [siboehm's CUDA Matrix Optimization](https://siboehm.com/articles/22/CUDA-MMM)
- [OpenCL SGEMM Tutorial](https://cnugteren.github.io/tutorial/pages/page1.html)
- [Cuda C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [metal_performance_testing by bkvogel](https://github.com/bkvogel/metal_performance_testing)
- [metal_flash_attention by philipturner](https://github.com/philipturner/metal-flash-attention/tree/main)

[^1]: https://www.cpu-monkey.com/en/cpu-apple_m2_8_gpu
[^2]: https://github.com/philipturner/metal-benchmarks
[^3]: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf