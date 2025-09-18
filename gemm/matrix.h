#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <random>

#include "Metal/MTLBuffer.hpp"
#include "Metal/MTLDevice.hpp"

struct MetalBufferDeleter
{
  void operator()(MTL::Buffer* buf) const { buf->release(); }
};

using BufferPtr = std::unique_ptr<MTL::Buffer, MetalBufferDeleter>;

class Matrix
{
public:
  size_t rows;
  size_t cols;

  Matrix()
      : rows(0)
      , cols(0)
      , data_(nullptr)
  {
  }

  Matrix(MTL::Device* device, size_t rows, size_t cols, float value = 0)
      : rows(rows)
      , cols(cols)
      , data_(nullptr)
  {
    if (device == nullptr) {
      throw std::runtime_error("Cannot create matrix with empty device");
    }

    data_ = BufferPtr(device->newBuffer(cols * rows * sizeof(float),
                                        MTL::ResourceStorageModeShared));
    this->fill(value);
  }

  Matrix(MTL::Device* device, size_t size, float value = 0)
      : Matrix(device, size, size, value)
  {
  }

  MTL::Buffer* device_data() { return data_.get(); }
  const MTL::Buffer* device_data() const { return data_.get(); }

  const float* host_data() const
  {
    return static_cast<float*>(data_.get()->contents());
  }
  float* data() { return static_cast<float*>(data_.get()->contents()); }

  float& operator[](size_t index) { return data()[index]; }
  const float& operator[](size_t index) const { return host_data()[index]; }

  void print()
  {
    std::cout << "Row: " << rows << ", Cols: " << cols << "\n";
    const float* raw_data = this->data();
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        std::cout << raw_data[i * cols + j] << " ";
      }
      std::cout << "\n";
    }
  }

  // create a random matrix
  static Matrix random(
      MTL::Device* device, float mu, float std, size_t rows, size_t cols)
  {
    static std::random_device rand_dev;
    static std::mt19937 generator(rand_dev());
    std::normal_distribution<float> distr(mu, std);

    // create matrix
    auto mat = Matrix(device, rows, cols, 0);
    float* raw_data = mat.data();
    for (size_t i = 0; i < rows * cols; i++) {
      raw_data[i] = distr(generator);
    }

    return mat;
  }

  // fill matrix with value
  void fill(float value)
  {
    float* raw_data = this->data();
    for (size_t i = 0; i < rows * cols; i++) {
      raw_data[i] = value;
    }
  }

private:
  BufferPtr data_;
};