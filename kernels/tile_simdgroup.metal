#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

struct MatmulParams
{
  uint M;
  uint N;
  uint K;
  uint32_t BLOCK_SIZE_X;
  uint32_t BLOCK_SIZE_Y;
};

// This kernel assumes a threadgroup size of 256 threads
// On Apple GPUs, 256 threads = 8 SIMD-groups.
// We will arrange these 8 SIMD-groups in a 4x2 grid,
// each computing an 8x8 tile.
// The full threadgroup computes a (4*8) x (2*8) = 32x16 tile of C.
kernel void matmul_tile_simdgroup(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const MatmulParams& params [[buffer(3)]],
    uint2 block_pos [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    // each SIMD-group computes an 8x8 tile
    const uint TILE_DIM = 8;

    // this is the threadgroup's base coordinate in the C matrix
    const uint tg_tile_row_base = block_pos.y * 32; // 4 tiles of 8
    const uint tg_tile_col_base = block_pos.x * 16; // 2 tiles of 8

    // this is this SIMD-group's local coordinate within the threadgroup tile
    const uint local_simd_row = (simd_id / 2) * TILE_DIM; // 0, 8, 16, 24
    const uint local_simd_col = (simd_id % 2) * TILE_DIM; // 0, 8

    // this is the final global C coordinate for this SIMD-group
    uint c_row = tg_tile_row_base + local_simd_row;
    uint c_col = tg_tile_col_base + local_simd_col;

    if (c_row >= params.M || c_col >= params.N) {
        return;
    }

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); 

    for (uint k = 0; k < params.K; k += TILE_DIM) {
        device const float* a_ptr = A + c_row * params.K + k;
        device const float* b_ptr = B + k * params.N + c_col;

        simdgroup_float8x8 a_tile;
        simdgroup_float8x8 b_tile;

        simdgroup_load(a_tile, a_ptr, params.K); 
        simdgroup_load(b_tile, b_ptr, params.N); 
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc); 
    }

    simdgroup_store(acc, C + c_row * params.N + c_col, params.N); 
}