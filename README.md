### Introduction

From Wikipedia [^1], Peak performance of my GPU (Apple's M2) is 3.6 TFLOPs = 3600 GFLOPs (10-core version). My GPU only has 8 cores, so by doing a simple calculation, the "real" peak performance should be 2880 GFLOPs (by Wikipedia, each core has 16 execution units so we will have about 360 GFLOPs each core, multiply by eight, we get 2880 GFLOPs). So let see if we can optimize our matrix optimization to (barely) reach that performance.

#### Set up

- You need to have `clang` compiler from Brew and `cmake, make`. So make sure you have installed all of them.
```bash
brew install llvm@20 cmake make
# Note: Currently llvm newest version (21) cannot work with MacOS properly
```

- Also remember to create `outputs` folder for `csv` logging.

- Then create a folder (for me I created `build` folder), go into that folder and type:
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
      -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
      -G "Unix Makefiles" \
      ..
```

- And finally run `make` to build. After a successful build, you can run the benchmark as below:
```bash
./bin/gemm <method_name>
# for example
# ./bin/gemm naive
```

> Valid names: "naive"

### References 

- https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
- https://github.com/leimao/CUDA-GEMM-Optimization/blob/main
- https://siboehm.com/articles/22/CUDA-MMM
- https://github.com/bkvogel/metal_performance_testing/tree/main

[^1]: en.wikipedia.org/wiki/Apple_M2