#include <metal_stdlib>

using namespace metal;

// We assume matrix is in row order
#define A(i, j) A[i * params.LDA + j]
#define B(i, j) B[i * params.LDB + j]
#define C(i, j) C[i * params.LDC + j]

struct MatmulParams
{
  uint M;
  uint N;
  uint K;
  uint LDA;
  uint LDB;
  uint LDC;
  float alpha;
  float beta;
  uint BLOCK_SIZE_X;
  uint BLOCK_SIZE_Y;
};

kernel void matmul_opt_3(device const float * A [[buffer(0)]],
                device const float * B [[buffer(1)]],
                device float * C       [[buffer(2)]],
                device const MatmulParams& params [[buffer(3)]],
                uint2 threadgroup_pos [[ threadgroup_position_in_grid ]],
                uint2 local_thread_idx [[ thread_position_in_threadgroup ]])
{
    // Note: be sure that this is set to the same value as "threads per group" in the calling code!
    // because we cannot directly set "template" for Metal code like CUDA
    const uint BLOCK_SIZE = 32;

    const uint M = params.M;
    const uint N = params.N;
    const uint K  = params.K;

    // Thread index
    const uint thread_x = local_thread_idx.x; // CUDA: threadIdx.x
    const uint thread_y = local_thread_idx.y; // CUDA: threadIdx.y (should be 1)

    // We will try to compute a sub matrix C
    const uint c_row = threadgroup_pos.x;
    const uint c_col = threadgroup_pos.y;

    // Allocate cache matrix
    threadgroup float As[BLOCK_SIZE * BLOCK_SIZE];
    threadgroup float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // Inner row and col
    const uint thread_col = thread_x % BLOCK_SIZE;
    const uint thread_row = thread_x / BLOCK_SIZE;

    // advance pointers to the starting positions
    A += c_row * BLOCK_SIZE * K;                     // row=cRow, col=0
    B += c_col * BLOCK_SIZE;                         // row=0, col=cCol
    C += c_row * BLOCK_SIZE * N + c_col * BLOCK_SIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
        As[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * K + thread_col];
        Bs[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];

        // block threads in this block until cache is fully populated
        threadgroup_barrier(mem_flags::mem_none);

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            tmp += As[thread_row * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + thread_col];
        }

        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        threadgroup_barrier(mem_flags::mem_none);
    }

    C[thread_row * N + thread_col] = params.alpha * tmp + params.beta * C[thread_row * N + thread_col];
}

/* CUDA Version (https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/3_kernel_shared_mem_blocking.cuh):
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}
*/