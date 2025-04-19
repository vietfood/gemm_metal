#include <metal_stdlib>

#define A(i, j) A[i * LDA + j]
#define B(i, j) B[i * LDB + j]
#define C(i, j) C[i * LDC + j]
#define Y(i) Y[(i) * INCY]

// Compute C = A * B + C
void adddot(constant uint& K,
            device const float * X,
            device const float * Y,
            constant uint& INCY, // stride of Y
            device float * GAMMA)
{
    // unroll loop 4 times
    for (uint p = 0; p < K; p += 4) {
        *GAMMA += X[p] * Y(p);
        *GAMMA += X[p + 1] * Y(p + 1);
        *GAMMA += X[p + 2] * Y(p + 2);
        *GAMMA += X[p + 3] * Y(p + 3);
    }
}

kernel void matmul_opt_2(device const float * A [[buffer(0)]],
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

    device const float* ptrA = &A(i, 0); // Start of row i in A
    device const float* ptrB = &B(0, j); // Start of column j in B
    device float* ptrC = &C(i, j); // Target element in C

    /* Update the C( i,j ) with the inner product of the ith row of A and the jth column of B */
    adddot(K, ptrA, ptrB, LDB, ptrC);
}