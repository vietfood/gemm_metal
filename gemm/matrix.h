#pragma once

#include "Metal/MTLResource.hpp"
#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "Metal/MTLBuffer.hpp"
#include "Metal/MTLDevice.hpp"

/** Matrix Class on GPU */
struct MetalBufferDeleter
{
  void operator()(MTL::Buffer* buf) const { buf->release(); }
};

using BufferPtr = std::unique_ptr<MTL::Buffer, MetalBufferDeleter>;

class DeviceMatrix
{
public:
  size_t rows;
  size_t cols;

  DeviceMatrix(MTL::Device* device, size_t rows, size_t cols)
      : rows(rows)
      , cols(cols)
      , data_(nullptr)
  {
    if (device == nullptr) {
      throw std::runtime_error("Cannot create matrix with empty device");
    }

    data_ = BufferPtr(device->newBuffer(cols * rows * sizeof(float),
                                        MTL::ResourceStorageModeShared));
  }

  const MTL::Buffer* data() const { return data_.get(); }
  MTL::Buffer* data() { return data_.get(); }

private:
  BufferPtr data_;
};

/** Matrix Class on CPU */
class HostMatrix
{
public:
  size_t rows;
  size_t cols;

  HostMatrix()
      : rows(0)
      , cols(0)
      , data_()
  {
  }

  HostMatrix(size_t rows, size_t cols, float value = 0)
      : rows(rows)
      , cols(cols)
      , data_(rows * cols, value)
  {
  }

  const float* data() const { return data_.data(); }
  float* data() { return data_.data(); }

  float& operator[](size_t index) { return data()[index]; }
  const float& operator[](size_t index) const { return data()[index]; }

  void print()
  {
    std::cout << "Row: " << rows << ", Cols: " << cols << "\n";
    const float* data = this->data();
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        std::cout << data[i * cols + j] << " ";
      }
      std::cout << "\n";
    }
  }

  // create a random matrix
  static HostMatrix random(float mu, float std, size_t rows, size_t cols)
  {
    static std::random_device rand_dev;
    static std::mt19937 generator(rand_dev());
    std::normal_distribution<float> distr(mu, std);

    // create matrix
    auto mat = HostMatrix(rows, cols, 0);
    float* raw_data = mat.data();
    for (size_t i = 0; i < rows * cols; i++) {
      raw_data[i] = distr(generator);
    }

    return mat;
  }

private:
  std::vector<float> data_;
};
