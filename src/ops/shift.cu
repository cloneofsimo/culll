
#include "big_cuops.h"

/// @brief If move_amount >= base / 2, it is considered negative shift.
/// @param batched_data_man 
/// @param amount 
/// @param B 
/// @param N 
/// @param M 
/// @param n 
/// @param logbase 
/// @param base 
/// @return 
__global__ void batchBigTensorKernelShift(lint* batched_data_man, lint* amount, lint B, lint N, lint M, lint n, lint logbase, lint base) {

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int pos = batch_idx * N * M + row_idx * M + col_idx;
    int is_leftshift = (amount[pos] >= base / 2) ? 1 : 0;
    int amount_abs = (is_leftshift) ? base - amount[pos] : amount[pos];

    int shiftamount_blk = amount_abs / logbase;
    int shiftamount_rem = amount_abs % logbase;

    int pos2 = batch_idx * N * M * n + row_idx * M * n + col_idx * n;

    if (batch_idx < B && row_idx < N && col_idx < M) {
        if (is_leftshift) {
            for (int i = n - shiftamount_blk - 1; i >= 1; i--) {
                batched_data_man[pos2 + shiftamount_blk + i] = (batched_data_man[pos2 + i] << shiftamount_rem) + (batched_data_man[pos2 + i - 1] >> (logbase - shiftamount_rem));
            }
            batched_data_man[pos2 + shiftamount_blk] = batched_data_man[pos2] << shiftamount_rem;
        }
        else {
            for (int i = 0; i < n - shiftamount_blk - 1; i++) {
                batched_data_man[pos2 + i] = (batched_data_man[pos2 + i + shiftamount_blk] >> shiftamount_rem) + (batched_data_man[pos2 + i + shiftamount_blk + 1] >> (logbase - shiftamount_rem));
            }
            batched_data_man[pos2 + n - shiftamount_blk - 1] = batched_data_man[pos2 + n - 1] >> shiftamount_rem;
        }
    }
}




__global__ void batchBigTensorKernelNormalizedShiftAmount(lint* batched_data_man, lint* amount_out, lint B, lint N, lint M, lint n, lint logbase, lint base) {

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int pos = batch_idx * N * M + row_idx * M + col_idx;
    int pos2 = batch_idx * N * M * n + row_idx * M * n + col_idx * n;


    lint sums = 0;
    lint lastval = 0;
    for(int i = n -1 ; i >= 0; i--){
        // adds 1 if the bit is 1 else 0
        lastval = batched_data_man[pos2 + i];
        if(lastval != 0){
            break;
        }
        else{
            sums += logbase;
        }
    }
    sums += logbase - __ffs(lastval) + 1;
    amount_out[pos] = sums;
}



