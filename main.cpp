#include <cassert>

#include "gemm/benchmark.h"

int main(int argc, char* argv[])
{
  BenchmarkMgr mgr;
  mgr.run_opt2();
  return 0;
}