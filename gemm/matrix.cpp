#include <iostream>
#include <random>

#include "Metal/MTLDevice.hpp"
#include "Metal/MTLResource.hpp"
#include "matrix.h"

Matrix::Matrix(MTL::Device* device, uint rows, uint cols)
    : rows(rows)
    , cols(cols)
{
  data = device->newBuffer(cols * rows * sizeof(float),
                           MTL::ResourceStorageModeShared);
}

void Matrix::free()
{
  data->release();
  cols = 0;
  rows = 0;
}

void Matrix::print()
{
  const float* raw_data = static_cast<float*>(data->contents());
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      std::cout << raw_data[i * cols + j] << " ";
    }
    std::cout << "\n";
  }
}

void Matrix::random_data(float mu, float std)
{
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::normal_distribution<float> distr(mu, std);

  float* raw_data = static_cast<float*>(data->contents());
  for (size_t i = 0; i < rows * cols; i++) {
    raw_data[i] = distr(generator);
  }
}

void Matrix::fill(float value)
{
  float* raw_data = static_cast<float*>(data->contents());
  for (size_t i = 0; i < rows * cols; i++) {
    raw_data[i] = value;
  }
}