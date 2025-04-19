#pragma once

constexpr int PFIRST = 256;
constexpr int PLAST = 5120;
constexpr int PINC = 128;

// https://github.com/philipturner/metal-benchmarks?tab=readme-ov-file#operations-per-second
constexpr float M2_GPU_GHZ = 1.398f;

constexpr int BENCHMARK_TIME = 50;
constexpr float EQUAL_EPSILON = 0.01f;