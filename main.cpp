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

  if (kernel_name == "opt_1") {
    mgr.run_opt1();
  }

  if (kernel_name == "opt_2") {
    mgr.run_opt2();
  }

  if (kernel_name == "opt_3") {
    mgr.run_opt3();
  }

  return 0;
}