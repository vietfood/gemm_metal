#include <metal_stdlib>

using namespace metal;

struct MatmulParams
{
  uint M;
  uint N;
  uint K;
  uint BLOCK_SIZE_X;
  uint BLOCK_SIZE_Y;
};

kernel void matmul_tile_threads(device const float * A [[buffer(0)]],
                                  device const float * B [[buffer(1)]],
                                  device float * C [[buffer(2)]],
                                  device const MatmulParams& params [[buffer(3)]],
                                  uint2 block_pos [[threadgroup_position_in_grid]],
                                  uint2 thread_pos [[thread_position_in_threadgroup]])
{
    // Tile size computed by one threadgroup
    constexpr uint TILE_M = 32; // 32 rows
    constexpr uint TILE_N = 32; // 32 cols

    // Inner dimension tile size
    constexpr uint TILE_K = 16;

    // Threadgroup size
    constexpr uint TG_M = 8; // 8 threads in M dim
    constexpr uint TG_N = 8; // 8 threads in N dim

    // Work per thread
    constexpr uint WPT_M = TILE_M / TG_M; // 4 rows per thread
    constexpr uint WPT_N = TILE_N / TG_N; // 4 cols per thread

    const uint thread_m = thread_pos.y; // 0..7 (TG_M-1)
    const uint thread_n = thread_pos.x; // 0..7 (TG_N-1)

    // Identify the top-left corner of the C tile this threadgroup works on
    const uint c_tile_m_start = block_pos.y * TILE_M;
    const uint c_tile_n_start = block_pos.x * TILE_N;

    // Threadgroup memory for tiles of A and B
    threadgroup float tileA[TILE_M][TILE_K];
    threadgroup float tileB[TILE_K][TILE_N];

    float C_reg[WPT_M][WPT_N] = {{0.0f}};

    for (uint t = 0; t < params.K; t += TILE_K) {
        // Each thread loads a 4x2 section of tileA and a 2x4 section of tileB
        // This ensures the 64 threads collectively load the 32x16 and 16x32 tiles.
        
        // Load for tileA (32x16)
        for (uint i = 0; i < WPT_M; ++i) { // 4 rows per thread
            uint a_row = c_tile_m_start + thread_m * WPT_M + i;
            uint a_col = t + thread_n; // Each thread loads 1 element per column in its group
            if (a_row < params.M && a_col < params.K)
                tileA[thread_m * WPT_M + i][thread_n] = A[a_row * params.K + a_col];
            else
                tileA[thread_m * WPT_M + i][thread_n] = 0.0f;
            // Repeat for the other half of K
             if (a_row < params.M && (a_col + TG_N) < params.K)
                tileA[thread_m * WPT_M + i][thread_n + TG_N] = A[a_row * params.K + a_col + TG_N];
            else
                tileA[thread_m * WPT_M + i][thread_n + TG_N] = 0.0f;
        }

        // Load for tileB (16x32)
        for (uint i = 0; i < WPT_N; ++i) { // 4 cols per thread
            uint b_row = t + thread_m;
            uint b_col = c_tile_n_start + thread_n * WPT_N + i;
            if (b_row < params.K && b_col < params.N)
                tileB[thread_m][thread_n * WPT_N + i] = B[b_row * params.N + b_col];
            else
                tileB[thread_m][thread_n * WPT_N + i] = 0.0f;
            // Repeat for other half of K
            if ((b_row + TG_M) < params.K && b_col < params.N)
                tileB[thread_m + TG_M][thread_n * WPT_N + i] = B[(b_row + TG_M) * params.N + b_col];
            else
                tileB[thread_m + TG_M][thread_n * WPT_N + i] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint k = 0; k < TILE_K; ++k) {
            for (uint m = 0; m < WPT_M; ++m) {
                float a_val = tileA[thread_m * WPT_M + m][k];
                for (uint n = 0; n < WPT_N; ++n) {
                    C_reg[m][n] += a_val * tileB[k][thread_n * WPT_N + n];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (uint m = 0; m < WPT_M; ++m) {
        for (uint n = 0; n < WPT_N; ++n) {
            uint c_row = c_tile_m_start + thread_m * WPT_M + m;
            uint c_col = c_tile_n_start + thread_n * WPT_N + n;
            if (c_row < params.M && c_col < params.N) {
                C[c_row * params.N + c_col] = C_reg[m][n];
            }
        }
    }
}