#pragma once

#include <string>
#include <unordered_map>

#include "gemm/kernel.h"
#include "gemm/metal_mgr.h"

static const std::string OPT_NAME[] = {"naive", "opt_1", "opt_2", "opt_3"};

class BenchmarkMgr
{
public:
  BenchmarkMgr();
  ~BenchmarkMgr();

  void run_naive();
  void run_opt1();
  void run_opt2();
  void run_opt3();

private:
  void run_common(const Matrix& A,
                  const Matrix& B,
                  Matrix& C,
                  Kernel* kernel,
                  MTL::Size thread_group_count,
                  MTL::Size thread_group_size);

  void run_common_time(const Matrix& A,
                       const Matrix& B,
                       Matrix& C,
                       Kernel* kernel,
                       MTL::Size thread_group_count,
                       MTL::Size thread_group_size);

  double run_multiples(const Matrix& A,
                       const Matrix& B,
                       Matrix& C,
                       Kernel* kernel,
                       MTL::Size thread_group_count,
                       MTL::Size thread_group_size);

  double get_run_time() const;

private:
  std::unique_ptr<MetalMgr> metal_;
  std::unordered_map<std::string, Kernel*> kernels_;
};