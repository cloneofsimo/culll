#include "hp_tensor.h"



__global__ void expMaximumSwapKernel(lint *batched_data_a_man, lint *batched_data_b_man, int* batched_data_a_expo, int *batched_data_b_expo, lint *batched_data_a_sign, lint *batched_data_b_sign, int B, int N, int M, int n){
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int pos = batch_idx * N * M + row_idx * M + col_idx;
    int pos2 = batch_idx * N * M * n + row_idx * M * n + col_idx * n;

    int expa = batched_data_a_expo[pos];
    int expb = batched_data_b_expo[pos];
    lint tmp;
    int toswap = (expa < expb) ? 1 : 0;
    if( toswap){
        for(int i = 0; i < n; i++){
            tmp = batched_data_b_man[pos2 + i];
            batched_data_b_man[pos2 + i] = batched_data_a_man[pos2 + i];
            batched_data_a_man[pos2 + i] = tmp;
        }

        tmp = batched_data_b_sign[pos];
        batched_data_b_sign[pos] = batched_data_a_sign[pos];
        batched_data_a_sign[pos] = tmp;

        tmp = batched_data_b_expo[pos];
        batched_data_b_expo[pos] = batched_data_a_expo[pos];
        batched_data_a_expo[pos] = tmp;

    }
}

void HPTensor::swap_exp_maximum(HPTensor* other){
    int B = this->B;
    int N = this->N;
    int M = this->M;
    int n = this->precision;

    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(B, N, M);

    expMaximumSwapKernel<<<dimGrid, dimBlock>>>(
        this->mantissa, other->mantissa, this->exponent, other->exponent, this->sign, other->sign, B, N, M, n);

}

__global__ void shiftExponentKernel(int *batched_data_expo, int *amount, lint B, lint N, lint M){
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int pos = batch_idx * N * M + row_idx * M + col_idx;

    batched_data_expo[pos] += amount[pos];
}

__global__ void shiftMantissaKernel(lint *batched_data_man, int *amount, lint B, lint N, lint M, lint n, lint logbase){
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int pos = batch_idx * N * M + row_idx * M + col_idx;
    int is_leftshift = (amount[pos] > 0) ? 1 : 0;
    int shiftamount_blk = abs(amount[pos]) / logbase;
    int shiftamount_rem = abs(amount[pos]) % logbase;

    int pos2 = batch_idx * N * M * n + row_idx * M * n + col_idx * n;

    if (batch_idx < B && row_idx < N && col_idx < M) {
        if(is_leftshift){
            for (int i = n - shiftamount_blk -1 ; i >= 1; i--) {
                batched_data_man[pos2 + shiftamount_blk + i] = (batched_data_man[pos2 + i] << shiftamount_rem) + (batched_data_man[pos2 + i - 1] >> (logbase - shiftamount_rem));
            }
            batched_data_man[pos2 + shiftamount_blk] = batched_data_man[pos2] << shiftamount_rem;
        }
        else{
            for(int i = 0; i < n - shiftamount_blk - 1; i++){
                batched_data_man[pos2 + i] = (batched_data_man[pos2 + i + shiftamount_blk] >> shiftamount_rem) + (batched_data_man[pos2 + i + shiftamount_blk + 1] >> (logbase - shiftamount_rem));
            }
            batched_data_man[pos2 + n - shiftamount_blk - 1] = batched_data_man[pos2 + n - 1] >> shiftamount_rem;
        }
    }
}

void HPTensor::reposition(int* amount){
    // shift the exponent by amount
    // shift the mantissa by amount
    // amount is B, N, M, 1 tensor
    // amount is the amount to shift by factor of 2.

    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(B, N, M);

    
    // shift mantissa
    shiftMantissaKernel<<<dimGrid, dimBlock>>>(this->mantissa, amount, B, N, M, precision, logbase);

    // shift exponent
    shiftExponentKernel<<<dimGrid, dimBlock>>>(this->exponent, amount, B, N, M);
    

}