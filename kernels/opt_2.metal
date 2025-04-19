#include <metal_stdlib>

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

kernel void matmul_opt_2(device const float * A [[buffer(0)]],
                device const float * B [[buffer(1)]],
                device float * C       [[buffer(2)]],
                device const MatmulParams& params [[buffer(3)]],
                uint2 threadgroup_pos [[ threadgroup_position_in_grid ]],
                uint2 local_thread_idx [[ thread_position_in_threadgroup ]])
{
    // Block index
    const uint block_x = threadgroup_pos.x; // CUDA: blockIdx.x
    const uint block_y = threadgroup_pos.y; // CUDA: blockIdx.y
    
    // Thread index
    const uint thread_x = local_thread_idx.x; // CUDA: threadIdx.x
    const uint thread_y = local_thread_idx.y; // CUDA: threadIdx.y

    // Calculate row and col
    const uint i = block_x * params.BLOCK_SIZE_X + (thread_x / params.BLOCK_SIZE_X); // col
    const uint j = block_y * params.BLOCK_SIZE_Y + (thread_x % params.BLOCK_SIZE_Y); // row 

    // Boundary check: Ensure this thread is calculating a valid element within C's bounds
    if (i < params.M && j < params.N)
    {
        float sum = 0.f;
        for (uint p = 0; p < params.K; ++p)
        {
            sum += A(i, p) * B(p, j);
        }
        C(i, j) = params.alpha * sum + params.beta * C(i, j);
    }
}