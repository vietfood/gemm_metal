#pragma once

#include <cstdint>

constexpr int X_THREADS_PER_GROUP = 32;  // Threads covering rows in a group
constexpr int Y_THREADS_PER_GROUP = 32;  // Threads covering columns in a group

constexpr int PFIRST = 256;
constexpr int PLAST = 3840;
constexpr int PINC = 128;

// https://github.com/philipturner/metal-benchmarks?tab=readme-ov-file#operations-per-second
constexpr float M2_GPU_GHZ = 1.398f;

constexpr int BENCHMARK_TIME = 20;
constexpr float EQUAL_EPSILON = 0.01f;

struct MatmulParams
{
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t LDA;
  uint32_t LDB;
  uint32_t LDC;
  float alpha;
  float beta;
  uint32_t BLOCK_SIZE_X;
  uint32_t BLOCK_SIZE_Y;
};