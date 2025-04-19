#include <metal_stdlib>

// We assume matrix is in row order
#define A(i, j) A[i * LDA + j]
#define B(i, j) B[i * LDB + j]
#define C(i, j) C[i * LDC + j]

kernel void matmul(device const float * A [[buffer(0)]],
                   device const float * B [[buffer(1)]],
                   device float * C       [[buffer(2)]],
                   constant uint& M       [[buffer(3)]], // Rows of A and C
                   constant uint& N       [[buffer(4)]], // Columns of B and C
                   constant uint& K       [[buffer(5)]], // Columns of A, Rows of B
                   constant uint& LDA     [[buffer(6)]], // Leading dimension of A
                   constant uint& LDB     [[buffer(7)]], // Leading dimension of B
                   constant uint& LDC     [[buffer(8)]], // Leading dimension of C
                   uint2 gid [[thread_position_in_grid]]) // Grid position (x -> col, y -> row)
{
    uint i = gid.y; // row
    uint j = gid.x; // col

    // Boundary check: Ensure this thread is calculating a valid element within C's bounds
    if (i >= M || j >= N) {
        return;
    }

    /* Update the C( i,j ) with the inner product of the ith row of A and the jth column of B */
    for (uint p = 0; p < K; ++p)
    {
        C(i, j) += A(i, p) * B(p, j);        
    }
}