#include <iostream>
#include <memory>
#include <stdexcept>

#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLComputePass.hpp"
#include "Metal/MTLTypes.hpp"
#include "benchmark.h"
#include "gemm/kernel.h"
#include "gemm/matrix.h"
#include "gemm/params.h"
#include "gemm/utils.h"

#define CREATE_DATA() \
  Matrix A(metal_->device, mat_size); \
  A.random_data(0.f, 1.f); \
  Matrix B(metal_->device, mat_size); \
  B.random_data(0.f, 1.f); \
  Matrix C(metal_->device, mat_size);

#define DELETE_DATA() \
  C.free(); \
  B.free(); \
  A.free();

#define WARM_UP() \
  run_common(A, B, C, kernel, thread_group_count, thread_group_size);

#define CHECK_CORRECTNESS() \
  Matrix D(metal_->device, mat_size); \
  matmul_cpu(A, B, D); \
  if (!equals(C, D)) { \
    throw std::runtime_error("Matrix multiplication on GPU isn't correct"); \
    D.free(); \
    C.free(); \
    B.free(); \
    A.free(); \
  }

#define RUN_BENCHMARK() \
  double time = \
      run_multiples(A, B, C, kernel, thread_group_count, thread_group_size); \
  double gflops = matmul_time_to_gflops(mat_size, mat_size, mat_size, time); \
  std::cout << "Matrix size " << mat_size << " tooks " << time / 1e3 \
            << " seconds average (after " << BENCHMARK_TIME \
            << " times) with gflops: " << gflops << "\n"; \
  kernel->csv_writer() << mat_size << time << gflops << endrow;

#define RUN() \
  WARM_UP() \
  if (mat_size == PFIRST) { \
    CHECK_CORRECTNESS() \
  } \
  RUN_BENCHMARK()

BenchmarkMgr::BenchmarkMgr()
{
  try {
    metal_ = std::make_unique<MetalMgr>();
    for (const auto& name : OPT_NAME) {
      kernels_[name] = new Kernel(name, metal_->device);
    }
  } catch (...) {
    throw;
  }
}

BenchmarkMgr::~BenchmarkMgr()
{
  for (auto& [name, kernel] : kernels_) {
    delete kernel;
  }
  kernels_.clear();
}

void BenchmarkMgr::run_naive()
{
  auto kernel = kernels_.at("naive");

  for (uint mat_size = PFIRST; mat_size < PLAST; mat_size += PINC) {
    CREATE_DATA();

    // calculate thread_group_count and size
    const uint x_thread_group_count =
        (C.rows + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP;
    const uint y_thread_group_count =
        (C.cols + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP;

    MTL::Size thread_group_count =
        MTL::Size::Make(x_thread_group_count, y_thread_group_count, 1);
    MTL::Size thread_group_size =
        MTL::Size::Make(X_THREADS_PER_GROUP, Y_THREADS_PER_GROUP, 1);

    RUN();
    DELETE_DATA()
  }
}

void BenchmarkMgr::run_opt1()
{
  auto kernel = kernels_.at("opt_1");

  for (uint mat_size = PFIRST; mat_size < PLAST; mat_size += PINC) {
    CREATE_DATA();

    // calculate thread_group_count and size
    const uint x_thread_group_count =
        (C.cols + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP;
    const uint y_thread_group_count =
        (C.rows + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP;

    MTL::Size thread_group_count =
        MTL::Size::Make(x_thread_group_count, y_thread_group_count, 1);
    MTL::Size thread_group_size =
        MTL::Size::Make(X_THREADS_PER_GROUP, Y_THREADS_PER_GROUP, 1);

    RUN();
    DELETE_DATA()
  }
}

void BenchmarkMgr::run_opt2()
{
  auto kernel = kernels_.at("opt_2");

  for (uint mat_size = PFIRST; mat_size < PLAST; mat_size += PINC) {
    CREATE_DATA();

    // calculate thread_group_count and size
    const uint x_thread_group_count =
        (C.rows + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP;
    const uint y_thread_group_count =
        (C.cols + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP;

    MTL::Size thread_group_count =
        MTL::Size::Make(x_thread_group_count, y_thread_group_count, 1);
    MTL::Size thread_group_size =
        MTL::Size::Make(X_THREADS_PER_GROUP * Y_THREADS_PER_GROUP, 1, 1);

    RUN();
    DELETE_DATA()
  }
}

void BenchmarkMgr::run_opt3()
{
  auto kernel = kernels_.at("opt_3");

  for (uint mat_size = PFIRST; mat_size < PLAST; mat_size += PINC) {
    CREATE_DATA();

    // calculate thread_group_count and size
    const uint x_thread_group_count =
        (C.rows + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP;
    const uint y_thread_group_count =
        (C.cols + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP;

    MTL::Size thread_group_count =
        MTL::Size::Make(x_thread_group_count, y_thread_group_count, 1);
    MTL::Size thread_group_size =
        MTL::Size::Make(X_THREADS_PER_GROUP * Y_THREADS_PER_GROUP, 1, 1);

    RUN();
    DELETE_DATA()
  }
}

double BenchmarkMgr::run_multiples(const Matrix& A,
                                   const Matrix& B,
                                   Matrix& C,
                                   Kernel* kernel,
                                   MTL::Size thread_group_count,
                                   MTL::Size thread_group_size)
{
  double total_time = 0;
  for (int time = 0; time < BENCHMARK_TIME; ++time) {
    run_common_time(A, B, C, kernel, thread_group_count, thread_group_size);
    total_time += get_run_time();
  }
  return total_time / static_cast<double>(BENCHMARK_TIME);
}

void BenchmarkMgr::run_common(const Matrix& A,
                              const Matrix& B,
                              Matrix& C,
                              Kernel* kernel,
                              MTL::Size thread_group_count,
                              MTL::Size thread_group_size)
{
  assert(A.cols == A.rows);  // assume square matrix
  assert(B.cols == B.rows);  // assume square matrix
  assert(A.cols == B.rows);

  assert(C.cols == B.cols);
  assert(C.rows == A.rows);

  uint M = C.rows;
  uint N = C.cols;
  uint K = A.cols;

  uint LDA = K;
  uint LDB = N;
  uint LDC = N;

  float alpha = 1.f;
  float beta = 1.f;

  MatmulParams params{M,
                      N,
                      K,
                      LDA,
                      LDB,
                      LDC,
                      alpha,
                      beta,
                      X_THREADS_PER_GROUP,
                      Y_THREADS_PER_GROUP};

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::CommandBuffer* cmd_buffer = metal_->cmd_queue->commandBuffer();
  assert(cmd_buffer != nullptr);

  MTL::ComputeCommandEncoder* compute_encoder =
      cmd_buffer->computeCommandEncoder();
  assert(compute_encoder != nullptr);

  compute_encoder->setComputePipelineState(kernel->pipeline());
  // move buffer to GPU
  compute_encoder->setBuffer(A.data, 0, 0);
  compute_encoder->setBuffer(B.data, 0, 1);
  compute_encoder->setBuffer(C.data, 0, 2);
  // move matrix params to GPU
  compute_encoder->setBytes(&params, sizeof(MatmulParams), 3);

  compute_encoder->dispatchThreadgroups(thread_group_count, thread_group_size);
  compute_encoder->endEncoding();

  cmd_buffer->commit();
  cmd_buffer->waitUntilCompleted();

  pool->release();
}

void BenchmarkMgr::run_common_time(const Matrix& A,
                                   const Matrix& B,
                                   Matrix& C,
                                   Kernel* kernel,
                                   MTL::Size thread_group_count,
                                   MTL::Size thread_group_size)
{
  assert(A.cols == A.rows);  // assume square matrix
  assert(B.cols == B.rows);  // assume square matrix
  assert(A.cols == B.rows);

  assert(C.cols == B.cols);
  assert(C.rows == A.rows);

  uint M = C.rows;
  uint N = C.cols;
  uint K = A.cols;

  uint LDA = K;
  uint LDB = N;
  uint LDC = N;

  float alpha = 1.f;
  float beta = 1.f;

  MatmulParams params{M,
                      N,
                      K,
                      LDA,
                      LDB,
                      LDC,
                      alpha,
                      beta,
                      X_THREADS_PER_GROUP,
                      Y_THREADS_PER_GROUP};

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::CommandBuffer* cmd_buffer = metal_->cmd_queue->commandBuffer();
  assert(cmd_buffer != nullptr);

  // https://developer.apple.com/videos/play/tech-talks/10001/
  MTL::ComputePassDescriptor* descriptor =
      MTL::ComputePassDescriptor::computePassDescriptor();
  auto sample_buffer_desc =
      static_cast<MTL::ComputePassSampleBufferAttachmentDescriptor*>(
          descriptor->sampleBufferAttachments()->object(0));
  sample_buffer_desc->setSampleBuffer(metal_->counter_buffer);
  sample_buffer_desc->setStartOfEncoderSampleIndex(0);
  sample_buffer_desc->setEndOfEncoderSampleIndex(1);

  MTL::ComputeCommandEncoder* compute_encoder =
      cmd_buffer->computeCommandEncoder(descriptor);
  assert(compute_encoder != nullptr);

  compute_encoder->setComputePipelineState(kernel->pipeline());
  // move buffer to GPU
  compute_encoder->setBuffer(A.data, 0, 0);
  compute_encoder->setBuffer(B.data, 0, 1);
  compute_encoder->setBuffer(C.data, 0, 2);
  // move matrix params to GPU
  compute_encoder->setBytes(&params, sizeof(MatmulParams), 3);

  compute_encoder->dispatchThreadgroups(thread_group_count, thread_group_size);
  compute_encoder->endEncoding();

  cmd_buffer->commit();
  cmd_buffer->waitUntilCompleted();

  pool->release();
}

double BenchmarkMgr::get_run_time() const
{
  auto counter_data =
      metal_->counter_buffer->resolveCounterRange(NS::Range::Make(0, 2));
  auto timestamps =
      static_cast<MTL::CounterResultTimestamp*>(counter_data->mutableBytes());
  uint64_t startTimeGPU = timestamps[0].timestamp;
  uint64_t endTimeGPU = timestamps[1].timestamp;
  double ms = static_cast<double>(endTimeGPU - startTimeGPU) / 1e6;
  return ms;
}