#pragma once

#include "Metal/MTLComputePipeline.hpp"
#include "Metal/MTLDevice.hpp"
#include "Metal/MTLLibrary.hpp"
#include "gemm/utils.h"

class Kernel
{
public:
  Kernel(const std::string& kernel_name, MTL::Device* device);
  ~Kernel();

  MTL::Library* library() const { return library_; }
  MTL::Function* function() const { return func_; }
  MTL::ComputePipelineState* pipeline() const { return pipeline_; }
  CSVWriter& csv_writer() const { return *writer_; }

private:
  MTL::Library* library_ = nullptr;
  MTL::Function* func_ = nullptr;
  MTL::ComputePipelineState* pipeline_ = nullptr;
  MTL::Device* device_ = nullptr;
  std::unique_ptr<CSVWriter> writer_ = nullptr;
};
