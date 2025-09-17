#include <cassert>

#include "gemm/benchmark.h"

int main(int argc, char* argv[])
{
  assert(argc == 2);
  std::string kernel_name = argv[1];
  BenchmarkMgr mgr;

  if (kernel_name == "naive") {
    mgr.run_naive();
  }

  return 0;
}