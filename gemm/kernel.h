#pragma once

#include "Metal/MTLComputePipeline.hpp"
#include "Metal/MTLDevice.hpp"
#include "Metal/MTLLibrary.hpp"
#include "gemm/utils.h"

struct KernelConfig
{
  // The dimensions of the threadgroup (block) in threads.
  size_t block_width = 32;
  size_t block_height = 32;

  // The dimensions of the output tile processed by a single threadgroup.
  // This is crucial for calculating the grid size correctly.
  size_t tile_width = 1;
  size_t tile_height = 1;
};

class Kernel
{
public:
  Kernel(const std::string& kernel_name, MTL::Device* device)
      : device_(device)
  {
    if (device_ == nullptr) {
      throw std::runtime_error("Device for kernel cannot be empty");
    }

    // setup kernel configuration
    if (kernel_name == "naive") {
      config_ = {.block_width = 32,
                 .block_height = 32,
                 .tile_width = 1,
                 .tile_height = 1};
    } else if (kernel_name == "tile_32") {
      config_ = {
          .block_width = 32,
          .block_height = 32,
          .tile_width = 1,
          .tile_height = 1,
      };
    } else if (kernel_name == "tile_16") {
      config_ = {
          .block_width = 16,
          .block_height = 16,
          .tile_width = 1,
          .tile_height = 1,
      };
    } else if (kernel_name == "tile_threads") {
      config_ = {
          .block_width = 8,
          .block_height = 8,
          .tile_width = 4,
          .tile_height = 4,
      };
    } else if (kernel_name == "tile_simdgroup") {
      config_ = {
          .block_width = 16,
          .block_height = 16,
          .tile_width = 1,
          .tile_height = 1,
      };
    } else {
      throw std::runtime_error("Unknown kernel name for configuration: "
                               + kernel_name);
    }

    NS::Error* error;

    // read kernel source
    std::string path = std::string(KERNEL_PATH) + kernel_name + ".metal";
    std::string src = read_file(path);
    auto metal_src =
        NS::String::string(src.c_str(), NS::StringEncoding::UTF8StringEncoding);

    // compile library
    library_ = device->newLibrary(metal_src, nullptr, &error);

    // check errors
    if (error != nullptr) {
      const char* msg = error->localizedDescription()->utf8String();
      throw std::runtime_error("Cannot create library because: "
                               + std::string(msg));
    }

    // create function
    auto str = NS::String::string(("matmul_" + kernel_name).c_str(),
                                  NS::ASCIIStringEncoding);
    func_ = library_->newFunction(str);

    // create pipeline for function
    pipeline_ = device->newComputePipelineState(func_, &error);

    if (error != nullptr) {
      const char* msg = error->localizedDescription()->utf8String();
      throw std::runtime_error("Cannot create library pipeline, error: "
                               + std::string(msg));
    }

    // create writer after everything
    writer_ = std::make_unique<CSVWriter>(kernel_name + ".csv");
  }

  ~Kernel()
  {
    func_->release();
    pipeline_->release();
    library_->release();
  }

  MTL::Library* library() const { return library_; }
  MTL::Function* function() const { return func_; }
  MTL::ComputePipelineState* pipeline() const { return pipeline_; }
  CSVWriter& writer() const { return *writer_; }
  const KernelConfig& config() const { return config_; }

private:
  MTL::Library* library_;
  MTL::Function* func_;
  MTL::ComputePipelineState* pipeline_;
  MTL::Device* device_;
  std::unique_ptr<CSVWriter> writer_;
  KernelConfig config_;
};