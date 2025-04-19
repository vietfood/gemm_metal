#include <memory>
#include <stdexcept>

#include "gemm/utils.h"
#include "kernel.h"

Kernel::Kernel(const std::string& kernel_name, MTL::Device* device)
    : device_(device)
{
  if (device_ == nullptr) {
    throw std::runtime_error("Device for kernel cannot be empty");
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

Kernel::~Kernel()
{
  func_->release();
  pipeline_->release();
  library_->release();
}