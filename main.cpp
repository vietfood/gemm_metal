#include <cassert>

#include "gemm/benchmark.h"

int main(int argc, char* argv[])
{
  assert(argc == 2);
  std::string kernel_name = argv[1];

  BenchmarkMgr mgr;
  mgr.run_benchmark_suite(kernel_name);

  return 0;
}