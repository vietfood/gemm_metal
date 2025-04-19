#include <cassert>
#include <iostream>
#include <stdexcept>

#include "Foundation/NSAutoreleasePool.hpp"
#include "Foundation/NSError.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLComputePass.hpp"
#include "Metal/MTLCounters.hpp"
#include "Metal/MTLResource.hpp"
#include "gemm/utils.h"
#include "metal_mgr.h"

MetalMgr::MetalMgr()
{
  device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    throw std::runtime_error("Cannot create device");
  }

  cmd_queue = device->newCommandQueue();
  if (cmd_queue == nullptr) {
    throw std::runtime_error("Cannot create common_queue");
  }

  // https://developer.apple.com/documentation/metal/creating-a-counter-sample-buffer-to-store-a-gpus-counter-data-during-a-pass?language=objc
  counter_set = get_counter_set(MTL::CommonCounterSetTimestamp, device);
  if (counter_set == nullptr) {
    throw std::runtime_error("Cannot create counter set");
  }

  sample_desc = MTL::CounterSampleBufferDescriptor::alloc()->init();
  sample_desc->setCounterSet(counter_set);
  sample_desc->setStorageMode(MTL::StorageModeShared);
  sample_desc->setSampleCount(2);

  NS::Error* error = nullptr;
  counter_buffer = device->newCounterSampleBuffer(sample_desc, &error);

  if (error != nullptr) {
    const char* msg = error->localizedDescription()->utf8String();
    throw std::runtime_error("Cannot create counter buffer because: "
                             + std::string(msg));
  }
}

MetalMgr::~MetalMgr()
{
  if (pipeline != nullptr) {
    pipeline->release();
  }

  if (func != nullptr) {
    func->release();
  }

  if (library != nullptr) {
    library->release();
  }

  counter_buffer->release();
  sample_desc->release();
  counter_set->release();
  cmd_queue->release();
  device->release();
}

void MetalMgr::set_kernel(const std::string& path, const std::string& func_name)
{
  if (library != nullptr) {
    library->release();
  }

  if (func != nullptr) {
    func->release();
  }

  if (pipeline != nullptr) {
    pipeline->release();
  }

  NS::Error* error;

  // read kernel source
  std::string src = read_file(std::string(KERNEL_PATH) + path);
  auto metal_src =
      NS::String::string(src.c_str(), NS::StringEncoding::UTF8StringEncoding);

  // compile library
  library = device->newLibrary(metal_src, nullptr, &error);

  // check errors
  if (error != nullptr) {
    const char* msg = error->localizedDescription()->utf8String();
    std::cerr << "Cannot create library with path: " << path
              << " because error: " << msg << "\n";
  }

  // create function
  auto str = NS::String::string(func_name.c_str(), NS::ASCIIStringEncoding);
  func = library->newFunction(str);

  // create pipeline for function
  pipeline = device->newComputePipelineState(func, &error);

  if (error != nullptr) {
    const char* msg = error->localizedDescription()->utf8String();
    std::cerr << "Cannot create library pipeline state because of error: "
              << msg << "\n";
  }
}

// same function with run but doesn't account time counting
void MetalMgr::warmup(const Matrix& A, const Matrix& B, Matrix& C)
{
  assert(A.cols == A.rows);  // assume square matrix
  assert(B.cols == B.rows);  // assume square matrix
  assert(A.cols == B.rows);

  assert(C.cols == B.cols);
  assert(C.rows = A.rows);

  uint M = C.rows;
  uint N = C.cols;
  uint K = A.cols;

  uint LDA = K;
  uint LDB = N;
  uint LDC = N;

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::CommandBuffer* cmd_buffer = cmd_queue->commandBuffer();
  assert(cmd_buffer != nullptr);

  MTL::ComputeCommandEncoder* compute_encoder =
      cmd_buffer->computeCommandEncoder();
  assert(compute_encoder != nullptr);

  compute_encoder->setComputePipelineState(pipeline);
  // move buffer to GPU
  compute_encoder->setBuffer(A.data, 0, 0);
  compute_encoder->setBuffer(B.data, 0, 1);
  compute_encoder->setBuffer(C.data, 0, 2);
  // move matrix size to GPU
  compute_encoder->setBytes(&M, sizeof(uint), 3);
  compute_encoder->setBytes(&N, sizeof(uint), 4);
  compute_encoder->setBytes(&K, sizeof(uint), 5);
  compute_encoder->setBytes(&LDA, sizeof(uint), 6);
  compute_encoder->setBytes(&LDB, sizeof(uint), 7);
  compute_encoder->setBytes(&LDC, sizeof(uint), 8);

  constexpr int x_threads_per_group = 8;  // Threads covering rows in a group
  constexpr int y_threads_per_group = 8;  // Threads covering columns in a group

  const int x_group_count = (N + x_threads_per_group - 1)
      / x_threads_per_group;  // loop columns first
  const int y_group_count =
      (M + x_threads_per_group - 1) / y_threads_per_group;  // then loop rows

  MTL::Size thread_group_count =
      MTL::Size::Make(x_group_count, y_group_count, 1);
  MTL::Size threadgroupSize =
      MTL::Size::Make(x_threads_per_group, y_threads_per_group, 1);
  compute_encoder->dispatchThreadgroups(thread_group_count, threadgroupSize);
  compute_encoder->endEncoding();

  cmd_buffer->commit();
  cmd_buffer->waitUntilCompleted();

  pool->release();
}

void MetalMgr::run(const Matrix& A, const Matrix& B, Matrix& C)
{
  assert(A.cols == A.rows);  // assume square matrix
  assert(B.cols == B.rows);  // assume square matrix
  assert(A.cols == B.rows);

  assert(C.cols == B.cols);
  assert(C.rows = A.rows);

  uint M = C.rows;
  uint N = C.cols;
  uint K = A.cols;

  uint LDA = K;
  uint LDB = N;
  uint LDC = N;

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::CommandBuffer* cmd_buffer = cmd_queue->commandBuffer();
  assert(cmd_buffer != nullptr);

  MTL::ComputePassDescriptor* descriptor =
      MTL::ComputePassDescriptor::computePassDescriptor();
  // https://developer.apple.com/videos/play/tech-talks/10001/
  auto sample_buffer_desc =
      static_cast<MTL::ComputePassSampleBufferAttachmentDescriptor*>(
          descriptor->sampleBufferAttachments()->object(0));
  sample_buffer_desc->setSampleBuffer(counter_buffer);
  sample_buffer_desc->setStartOfEncoderSampleIndex(0);
  sample_buffer_desc->setEndOfEncoderSampleIndex(1);

  MTL::ComputeCommandEncoder* compute_encoder =
      cmd_buffer->computeCommandEncoder(descriptor);
  assert(compute_encoder != nullptr);

  compute_encoder->setComputePipelineState(pipeline);
  // move buffer to GPU
  compute_encoder->setBuffer(A.data, 0, 0);
  compute_encoder->setBuffer(B.data, 0, 1);
  compute_encoder->setBuffer(C.data, 0, 2);
  // move matrix size to GPU
  compute_encoder->setBytes(&M, sizeof(uint), 3);
  compute_encoder->setBytes(&N, sizeof(uint), 4);
  compute_encoder->setBytes(&K, sizeof(uint), 5);
  compute_encoder->setBytes(&LDA, sizeof(uint), 6);
  compute_encoder->setBytes(&LDB, sizeof(uint), 7);
  compute_encoder->setBytes(&LDC, sizeof(uint), 8);

  constexpr int x_threads_per_group = 8;  // Threads covering rows in a group
  constexpr int y_threads_per_group = 8;  // Threads covering columns in a group

  const int x_group_count = (N + x_threads_per_group - 1)
      / x_threads_per_group;  // loop columns first
  const int y_group_count =
      (M + y_threads_per_group - 1) / y_threads_per_group;  // then loop rows

  MTL::Size thread_group_count =
      MTL::Size::Make(x_group_count, y_group_count, 1);
  MTL::Size threadgroupSize =
      MTL::Size::Make(x_threads_per_group, y_threads_per_group, 1);
  compute_encoder->dispatchThreadgroups(thread_group_count, threadgroupSize);
  compute_encoder->endEncoding();

  cmd_buffer->commit();
  cmd_buffer->waitUntilCompleted();

  pool->release();
}

double MetalMgr::get_run_time() const
{
  auto counter_data =
      counter_buffer->resolveCounterRange(NS::Range::Make(0, 2));
  auto timestamps =
      static_cast<MTL::CounterResultTimestamp*>(counter_data->mutableBytes());
  uint64_t startTimeGPU = timestamps[0].timestamp;
  uint64_t endTimeGPU = timestamps[1].timestamp;
  double ms = static_cast<double>(endTimeGPU - startTimeGPU) / 1e6;
  return ms;
}