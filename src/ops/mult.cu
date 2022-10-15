#include <gpu_utils.h>
#include <big_cuops.h>

/// @brief Naive batchwise bigint multiplciation kernel.
/// @param batched_data_a B x N x M x n
/// @param batched_data_b B x N x M x n
/// @param output_data B x N x M x n
/// @param B
/// @param N
/// @param M
/// @param n is the number of bits allocated.
/// @param lens is number of bits to actually use. In our
/// case, typically lens <= n / 2.
/// @param base
/// @return
__global__ void batchBigTensorKernelOffsetMult(lint *batched_data_a,
                                          lint *batched_data_b,
                                          lint *output_data, lint B, lint N,
                                          lint M, lint n, lint a_start,
                                          lint b_start, lint out_start,
                                          lint a_n, lint b_n, lint base = 10){

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;
    // use for loops to iterate over n

    lint sum = 0;
    lint overflow = 0;

    int pos = batch_idx * N * M * n + row_idx * M * n + col_idx * n;

    for (int i = 0; i < a_n; i++) {
        for (int j = 0; j < b_n; j++) {
            lint a = batched_data_a[pos + i + a_start];
            lint b = batched_data_b[pos + j + b_start];
            sum = a * b + overflow;
            if (sum >= base) {
                overflow = sum / base;
                sum %= base;
            } else {
                overflow = 0;
            }
            //
            output_data[pos + i + j + out_start] += sum;
        }
        output_data[pos + i + b_n + out_start] += overflow;
        overflow = 0;
    }
    for(int i = out_start; i < out_start + a_n + b_n; i++){
        if(output_data[pos + i] >= base){
            output_data[pos + i + 1] += output_data[pos + i] / base;
            output_data[pos + i] %= base;
        }
    }
}


void batchBigTensorMult(int B, int N, int M, int n1, int n2, int n3, lint *data_a,lint *data_b, lint *data_c, int base = 10, int i_to_gpu = 1, int o_to_cpu = 1, int mode = 0){
    
    lint *gpu_ptr_a;
    lint *gpu_ptr_b;
    lint *gpu_ptr_c;

    int data_a_size = B * N * M * n1;
    int data_b_size = B * N * M * n2;
    int data_c_size = B * N * M * n3;
    
    if(i_to_gpu){
        gpuErrchk(cudaMalloc(&gpu_ptr_a, data_a_size * sizeof(lint)));
        gpuErrchk(cudaMalloc(&gpu_ptr_b, data_b_size * sizeof(lint)));
        gpuErrchk(cudaMalloc(&gpu_ptr_c, data_c_size * sizeof(lint)));

        gpuErrchk(cudaMemcpy(gpu_ptr_a, data_a, data_a_size * sizeof(lint),
                            cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gpu_ptr_b, data_b, data_b_size * sizeof(lint),
                            cudaMemcpyHostToDevice));
    }
    else{
        // assume that given pointers are on gpu.
        gpu_ptr_a = data_a;
        gpu_ptr_b = data_b;
        gpu_ptr_c = data_c;
    }


    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(B, N, M);
    if (mode == 0) {
        batchBigTensorKernelOffsetMult<<<dimGrid, dimBlock>>>(
            gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, B, N, M, n3, 0, 0, 0, n1, n2, base);
    } else {
        std::cout << "Not implemented yet" << std::endl;
    }

    if(o_to_cpu){
        
        gpuErrchk(cudaMemcpy(data_c, gpu_ptr_c, data_c_size * sizeof(lint),
                            cudaMemcpyDeviceToHost));
        cudaFree(gpu_ptr_a);
        cudaFree(gpu_ptr_b);
        cudaFree(gpu_ptr_c);
        
    }
}



void batchBigTensorMultWrapper(pybind11::array_t<lint> batched_data_a,
                                pybind11::array_t<lint> batched_data_b,
                                pybind11::array_t<lint> output_data,
                                int mode = 1, int verbose = 0, int base = 10) {
    pybind11::buffer_info ha = batched_data_a.request();
    pybind11::buffer_info hb = batched_data_b.request();
    pybind11::buffer_info hc = output_data.request();

    int B, N, M, n1, n2, n3;

    B = ha.shape[0];
    N = ha.shape[1];
    M = ha.shape[2];
    n1 = ha.shape[3];
    n2 = hb.shape[3];
    n3 = hc.shape[3];

    threedim_checker(ha, "batched_data_a", verbose);
    threedim_checker(hb, "batched_data_b", verbose, B, N, M);
    threedim_checker(hc, "output_data", verbose, B, N, M);
    
    lint *data_a = (lint *)ha.ptr;
    lint *data_b = (lint *)hb.ptr;
    lint *data_c = (lint *)hc.ptr;

    batchBigTensorMult(B, N, M, n1, n2, n3, data_a, data_b, data_c, 10, 1, 1, 0);
    
}
