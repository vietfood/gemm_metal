#include <cassert>
#include <iostream>

#include "gemm/matrix.h"
#include "gemm/metal_mgr.h"
#include "gemm/params.h"
#include "gemm/utils.h"

int main(int argc, char* argv[])
{
  assert(argc == 2);

  // second is kernel name
  std::string kernel_name = argv[1];

  MetalMgr mgr;
  CSVWriter writer(kernel_name + ".csv");

  for (uint mat_size = PFIRST; mat_size < PLAST; mat_size += PINC) {
    Matrix A(mgr.device, mat_size, mat_size);
    A.random_data(0.f, 1.f);

    Matrix B(mgr.device, mat_size, mat_size);
    B.random_data(0.f, 1.f);

    Matrix C(mgr.device, mat_size, mat_size);
    Matrix D(mgr.device, mat_size, mat_size);

    // run naive version first
    mgr.set_kernel("naive.metal", "matmul");
    mgr.warmup(A, B, D);

    // then change to current kernel and run to compare correctness
    mgr.set_kernel(kernel_name + ".metal",
                   std::string("matmul_") + kernel_name);
    mgr.warmup(A, B, C);

    if (!equals(C, D)) {
      std::cerr << "Matrix multiplication on GPU isn't correct";
      D.free();
      C.free();
      B.free();
      A.free();
      return -1;
    }
    // free temporary matrix D
    D.free();

    // run the benchmark if we can make sure the correctness
    auto time = benchmark(&mgr, A, B, C);
    auto gflops = matmul_time_to_gflops(mat_size, mat_size, mat_size, time);

    std::cout << "Matrix size " << mat_size << " tooks " << time / 1e3
              << " seconds average (after " << BENCHMARK_TIME
              << " times) with gflops: " << gflops << "\n";
    writer << mat_size << time << gflops << endrow;

    // free all matrix
    C.free();
    B.free();
    A.free();
  }

  return 0;
}