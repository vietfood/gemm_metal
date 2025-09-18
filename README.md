# A journey to ~2.88 TFLOPs on my M2 MacBook

This is a diary of my quest to write a face-meltingly fast matrix multiplication (GEMM) kernel on my Apple M2 laptop. Since I don't have an NVIDIA card lying around, I'm using Apple's **Metal** API instead of CUDA. The core ideas are the same: making GEMM as fast as possible. 

The theoretical FP32 peak of a 10-core M2 is ~3.6 TFLOPs. Mine's an 8-core, so is **~2.88 TFLOPs**. Can we even get close? Let's find out.

This repo is for anyone who wants to learn GPU optimization but is tired of vendor-locked CUDA tutorials. The code is (a little bit) clean, heavily commented (where it matters), and built with modern C++ so we don't leak memory all over the place (and shoot our foot).

## How fast are we so far?

| Kernel          | Best Performance (GFLOPS) | Notes                                           |
|-----------------|---------------------------|-------------------------------------------------|
| `naive`         | ~160                      | It's a start! One thread, one MAC. So pure.     |

## Get it running

You'll need a Mac with an M-series chip. Sorry, no Windows or Linux love here, this is a Metal party.

**1. Install the tools of the trade.**

If you don't have Homebrew, get it. Then:
```bash
brew install cmake make llvm@20
```
We need `llvm`'s clang because Apple's default one can be a bit... quirky. Note that `llvm` newest version (21) cannot work on MacOS properly so we choose version 20.

**2. Build the thing.**

Pop open a terminal and run these:
```bash
# Make a home for the benchmark results
mkdir -p outputs

# Let CMake do its magic. This points to the new clang we just installed.
cmake -S . -B build -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang \
      -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++

# Fire the lasers!
cmake --build build
```
If that all worked, you'll have a shiny new executable at `build/bin/gemm`.

## Run the benchmark!

Time for the moment of truth. Pit your kernel against a whole gauntlet of matrix shapes designed to stress it out.

```bash
# It's easy, just tell it which kernel to run
./build/bin/gemm naive
```

The program will spit out performance numbers to your console and also save a detailed `.csv` file in the `outputs` folder. This is the good stuff you can use to make pretty graphs in Python.

**Available kernels:**
- `naive`: The humble beginning. 

## The Grand Plan (aka The Optimization Checklist)

This project is structured so you can follow the optimization journey step-by-step. Each new kernel will be a separate file, building on the lessons of the last.

- [x] **Chapter 0: The Setup.** Build a solid, memory-safe C++ framework with a real benchmark harness. No segfaults allowed.
- [ ] **Chapter 1: The Tiling.** The first real optimization. Use that sweet, sweet threadgroup memory (`__threadgroup` in Metal, `__shared__` in CUDA) to stop hitting DRAM so much. This is where we should see the first big performance jump.
<!-- -   [ ] **Chapter 2: More Work, Less Laziness.** Make each thread compute more than one output element. This hides instruction latency and is great for register reuse.
-   [ ] **Chapter 3: SIMD-ify Everything.** Use Metal's vector types (`float4`, `half4`) to get more math done per clock cycle.
-   [ ] **Chapter 4: The Final Boss.** Memory coalescing, bank conflicts, and other arcane horrors. -->

## Resources

- [Leimao's CUDA GEMM Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [siboehm's CUDA MMM](https://siboehm.com/articles/22/CUDA-MMM)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)