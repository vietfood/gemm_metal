### Introduction

From Wikipedia [^1], Peak performance of my GPU (Apple's M2) is 3.6 TFLOPs = 3600 GFLOPs (10-core version). My GPU only has 8 cores, so by doing a simple calculation, the "real" peak performance should be 2880 GFLOPs (by Wikipedia, each core has 16 execution units so we will have about 360 GFLOPs each core, multiply by eight, we get 2880 GFLOPs). So let see if we can optimize our matrix optimization to (barely) reach that performance.

#### Set up

We will use $32 \times 32$ threads.

#### Naive method

With the naive method, we got around 180-200 GLFOPs, which is really far from our peak performance.

### References 

- https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
- https://github.com/leimao/CUDA-GEMM-Optimization/blob/main
- https://siboehm.com/articles/22/CUDA-MMM
- https://github.com/bkvogel/metal_performance_testing/tree/main


[^1]: en.wikipedia.org/wiki/Apple_M2