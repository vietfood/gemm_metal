#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLComputePass.hpp"
#include "Metal/MTLTypes.hpp"
#include "benchmark.h"
#include "gemm/kernel.h"
#include "gemm/matrix.h"
#include "gemm/params.h"
#include "gemm/utils.h"

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
  run_benchmark_suite("naive");
}

void BenchmarkMgr::run_benchmark_suite(const std::string& kernel_name)
{
  auto kernel_iter = kernels_.find(kernel_name);
  if (kernel_iter == kernels_.end()) {
    throw std::runtime_error("Kernel not found: " + kernel_name);
  }

  Kernel* kernel = kernel_iter->second;
  std::cout << "\n--- Running Benchmark Suite for Kernel: " << kernel_name
            << " ---\n";

  // Write headers to the CSV file
  kernel->writer() << "M" << "N" << "K" << "Time (ms)" << "GFLOPS" << endrow;

  for (const auto& shape : BENCHMARK_SHAPES) {
    const size_t M = shape.M;
    const size_t N = shape.N;
    const size_t K = shape.K;

    std::cout << "Benchmarking shape (M, N, K): (" << M << ", " << N << ", "
              << K << ")... ";

    Matrix A = Matrix::random(metal_->device, 0.f, 1.f, M, K);
    Matrix B = Matrix::random(metal_->device, 0.f, 1.f, K, N);
    Matrix C(metal_->device, M, N);

    // We use thread size 32 as CUDA
    MTL::Size block = MTL::Size::Make(32, 32, 1);
    MTL::Size grid = MTL::Size::Make((N + 32 - 1) / 32, (M + 32 - 1) / 32, 1);

    // First, warm up
    start_kernel(A, B, C, kernel, grid, block, false);

    // Then, check correctness
    if (M <= 1024 && N <= 1024 && K <= 1024) {  // Example condition
      Matrix D(metal_->device, M, N);  // CPU matrix doesn't need a device
      matmul_cpu(A, B, D);
      if (!equals(C, D)) {
        throw std::runtime_error("FAILED correctness check for shape ("
                                 + std::to_string(M) + "," + std::to_string(N)
                                 + "," + std::to_string(K) + ")");
      }
    }

    // Timed benchmark runs
    double time_ms = run_multiples(A, B, C, kernel, grid, block);
    double gflops = matmul_time_to_gflops(M, N, K, time_ms);

    std::cout << "Time: " << time_ms << " ms, GFLOPS: " << gflops << "\n";
    kernel->writer() << M << N << K << time_ms << gflops << endrow;
  }
}

void BenchmarkMgr::start_kernel(const Matrix& A,
                                const Matrix& B,
                                Matrix& C,
                                Kernel* kernel,
                                MTL::Size grid_size,
                                MTL::Size block_size,
                                bool timer)
{
  assert(A.cols == B.rows);
  assert(C.cols == B.cols);
  assert(C.rows == A.rows);

  uint M = C.rows;
  uint N = C.cols;
  uint K = A.cols;

  float alpha = 1.f;
  float beta = 1.f;

  MatmulParams params{M,
                      N,
                      K,
                      alpha,
                      beta,
                      static_cast<uint32_t>(block_size.width),
                      static_cast<uint32_t>(block_size.height)};

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::CommandBuffer* cmd_buffer = metal_->cmd_queue->commandBuffer();
  if (cmd_buffer == nullptr) {
    throw std::runtime_error(
        "Cannot run kernel because command buffer is null");
  }

  MTL::ComputePassDescriptor* descriptor = nullptr;
  if (timer) {
    // https://developer.apple.com/videos/play/tech-talks/10001/
    descriptor = MTL::ComputePassDescriptor::computePassDescriptor();
    auto sample_buffer_desc =
        static_cast<MTL::ComputePassSampleBufferAttachmentDescriptor*>(
            descriptor->sampleBufferAttachments()->object(0));
    sample_buffer_desc->setSampleBuffer(metal_->counter_buffer);
    sample_buffer_desc->setStartOfEncoderSampleIndex(0);
    sample_buffer_desc->setEndOfEncoderSampleIndex(1);
  }

  MTL::ComputeCommandEncoder* compute_encoder =
      cmd_buffer->computeCommandEncoder(descriptor);
  if (compute_encoder == nullptr) {
    throw std::runtime_error(
        "Cannot run kernel because compute encoder is null");
  }
  compute_encoder->setComputePipelineState(kernel->pipeline());

  // create compute pipeline
  // move buffer to GPU
  compute_encoder->setBuffer(A.device_data(), 0, 0);
  compute_encoder->setBuffer(B.device_data(), 0, 1);
  compute_encoder->setBuffer(C.device_data(), 0, 2);

  // move matrix params to GPU
  compute_encoder->setBytes(&params, sizeof(MatmulParams), 3);

  // create threads
  compute_encoder->dispatchThreadgroups(grid_size, block_size);
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

double BenchmarkMgr::run_multiples(const Matrix& A,
                                   const Matrix& B,
                                   Matrix& C,
                                   Kernel* kernel,
                                   MTL::Size grid_size,
                                   MTL::Size block_size)
{
  double total_time = 0;
  for (int time = 0; time < BENCHMARK_TIME; ++time) {
    start_kernel(A, B, C, kernel, grid_size, block_size, true);
    total_time += get_run_time();
  }
  return total_time / static_cast<double>(BENCHMARK_TIME);
}