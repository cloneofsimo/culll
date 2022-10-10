
#include <big_cuops.h>
#include <gpu_utils.h>


using lint = unsigned int;


/// @brief D&C Karatsuba multiplication, all pointers have
/// to be on GPU.
/// @param batch_a
/// @param batch_b
/// @param batch_out
/// @param B
/// @param N
/// @param M
/// @param n
/// @param lens
/// @param base
void batchKaratsuba(lint* batch_x, lint* batch_y, lint* batch_out, lint B,
    lint N, lint M, lint n, lint a_start, lint b_start,
    lint out_start, lint a_len, lint b_len, lint base) {

    // where to clear gpu memeory?

    lint n_lower = n / 2;
    lint n_upper = n - n_lower;

    if (a_len < 4 || b_len < 4) {
        // naive multiplication.
        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(B, N, M);

        batchLongTensorOffsetMult << <dimGrid, dimBlock >> > (
            batch_x, batch_y, batch_out, B, N, M, n, a_start0, b_start,
            out_start, a_len, b_len, base);

        return;
    }
    else {

        lint* ac, * bd, * ad_plus_bc, * a_plus_b, * c_plus_d;

        // a : upper of x
        // b : lower of x
        // c : upper of y
        // d : lower of y

        gpuErrchk(cudaMalloc(&ac, sizeof(lint) * B * N * M * n));
        gpuErrchk(cudaMalloc(&bd, sizeof(lint) * B * N * M * n));
        gpuErrchk(cudaMalloc(&ad_plus_bc, sizeof(lint) * B * N * M * n));
        gpuErrchk(cudaMalloc(&a_plus_b, sizeof(lint) * B * N * M * n));
        gpuErrchk(cudaMalloc(&c_plus_d, sizeof(lint) * B * N * M * n));

        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(B, N, M);

        // ac
        batchKaratsuba(batch_x, batch_y, ac, B, N, M, n,
            a_start + n_lower, b_start + n_lower, 0, n_upper,
            n_upper, base);

        // bd
        batchKaratsuba(batch_x, batch_y, bd, B, N, M, n, a_start,
            b_start, 0, n_lower, n_lower, base);

        // a + b
        batchLongTensorOffsetAdd(batch_x, batch_x, a_plus_b, B, N, M, n,
            a_start + n_lower, a_start, 0, n_upper, base);

        // c + d
        batchLongTensorOffsetAdd(batch_y, batch_y, c_plus_d, B, N, M, n,
            b_start + n_lower, b_start, 0, n_lower, base);

        // (a + b) * (c + d)
        batchKaratsuba(a_plus_b, c_plus_d, ad_plus_bc, B, N, M, n, 0, 0, 0,
            n, n, base);

        // negate ac, bd
        batchLongTensorNegate(ac, B, N, M, n, base);
        batchLongTensorNegate(bd, B, N, M, n, base);

        // ad_plus_bc = (a + b) * (c + d) - ac - bd
        batchLongTensorOffsetAdd(ad_plus_bc, ac, ad_plus_bc, B, N, M, n, 0, 0, 0,
            n, n, base);

        batchLongTensorOffsetAdd(ad_plus_bc, bd, ad_plus_bc, B, N, M, n, 0, 0, 0,
            n, n, base);

        // TODO : Make simpler operations... By refactoring mat, add in class-driven ways...





    }
}