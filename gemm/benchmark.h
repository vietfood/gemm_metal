#pragma once

#include <string>
#include <unordered_map>

#include "gemm/kernel.h"
#include "gemm/matrix.h"
#include "gemm/metal_mgr.h"

static const std::string OPT_NAME[] = {"naive"};

class BenchmarkMgr
{
public:
  BenchmarkMgr();
  ~BenchmarkMgr();

  void run_naive();

private:
  void run_benchmark_suite(const std::string& kernel_name);

  void start_kernel(const Matrix& A,
                    const Matrix& B,
                    Matrix& C,
                    Kernel* kernel,
                    MTL::Size grid_size,
                    MTL::Size block_size,
                    bool timer = false);

  double run_multiples(const Matrix& A,
                       const Matrix& B,
                       Matrix& C,
                       Kernel* kernel,
                       MTL::Size grid_size,
                       MTL::Size block_size);

  double get_run_time() const;

private:
  std::unique_ptr<MetalMgr> metal_;
  std::unordered_map<std::string, Kernel*> kernels_;
};