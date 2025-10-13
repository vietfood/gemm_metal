#include <metal_stdlib>

using namespace metal;

struct MatmulParams
{
  uint M;
  uint N;
  uint K;
  float alpha;
  float beta;
  uint BLOCK_SIZE_X;
  uint BLOCK_SIZE_Y;
};

kernel void matmul_tile_16(device const float * A [[buffer(0)]],
                         device const float * B [[buffer(1)]],
                         device float * C [[buffer(2)]],
                         device const MatmulParams& params [[buffer(3)]],
                         uint2 block_pos [[ threadgroup_position_in_grid ]],
                         uint2 thread_pos [[ thread_position_in_threadgroup ]])
{
    // create tile block
    constexpr uint TILE_SIZE = 16;
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    // block index
    const uint block_x = block_pos.x; // CUDA: blockIdx.x
    const uint block_y = block_pos.y; // CUDA: blockIdx.y
    
    // thread index
    const uint thread_x = thread_pos.x; // CUDA: threadIdx.x
    const uint thread_y = thread_pos.y; // CUDA: threadIdx.y

    // calculate row and col
    uint row = block_y * TILE_SIZE + thread_y;
    uint col = block_x * TILE_SIZE + thread_x;

    float sum = 0.0f;
    for (uint t = 0; t < (params.K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        uint tiledColA = t * TILE_SIZE + thread_x;
        uint tiledRowB = t * TILE_SIZE + thread_y;

        // load elements to tile
        if (row < params.M && tiledColA < params.K)
            tileA[thread_y][thread_x] = A[row * params.K + tiledColA];
        else
            tileA[thread_y][thread_x] = 0.0f;
        if (tiledRowB < params.K && col < params.N)
            tileB[thread_y][thread_x] = B[tiledRowB * params.N + col];
        else
            tileB[thread_y][thread_x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // fast matmul on tile
#pragma unroll
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[thread_y][k] * tileB[k][thread_x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < params.M && col < params.N) {
        C[row * params.N + col] = params.alpha * sum + params.beta * C[row * params.N + col];
    }
}