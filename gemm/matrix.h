#pragma once

#include <cstddef>

#include "Metal/MTLBuffer.hpp"
#include "gemm/params.h"

struct Matrix
{
  uint rows;
  uint cols;
  MTL::Buffer* data;

  Matrix()
      : rows(0)
      , cols(0)
      , data(nullptr)
  {
  }

  Matrix(MTL::Device* device, uint rows, uint cols);

  float* host_data() const { return static_cast<float*>(data->contents()); }
  float& operator[](uint index) { return host_data()[index]; }
  const float& operator[](uint index) const { return host_data()[index]; }

  void free();
  void print();
  // random based on normal distribution
  void random_data(float mu, float std);
  void fill(float value);
};

inline bool equals(const Matrix& A, const Matrix& B)
{
  assert(A.cols == B.cols);
  assert(B.rows == A.rows);

  uint rows = A.rows;
  uint cols = A.cols;

  for (size_t i = 0; i < rows * cols; ++i) {
    float a = A[i];
    float b = B[i];
    bool is_approx_equal = fabs(a - b)
        <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EQUAL_EPSILON);
    if (!is_approx_equal) {
      return false;
    }
  }

  return true;
}