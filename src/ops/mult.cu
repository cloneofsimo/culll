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
__global__ void batchLongTensorOffsetMult(lint *batched_data_a,
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

void batchLongTensorMultWrapper(pybind11::array_t<lint> batched_data_a,
                                pybind11::array_t<lint> batched_data_b,
                                pybind11::array_t<lint> output_data,
                                int mode = 1, int verbose = 0, int base = 10) {
    pybind11::buffer_info ha = batched_data_a.request();
    pybind11::buffer_info hb = batched_data_b.request();
    pybind11::buffer_info hc = output_data.request();

    if (ha.ndim != 4) {
        std::stringstream strstr;
        strstr << "ha.ndim != 4" << std::endl;
        strstr << "ha.ndim: " << ha.ndim << std::endl;
        throw std::runtime_error(strstr.str());
    }

    if (hb.ndim != 4) {
        std::stringstream strstr;
        strstr << "hb.ndim != 4" << std::endl;
        strstr << "hb.ndim: " << hb.ndim << std::endl;
        throw std::runtime_error(strstr.str());
    }

    if (hc.ndim != 4) {
        std::stringstream strstr;
        strstr << "hc.ndim != 4" << std::endl;
        strstr << "hc.ndim: " << hc.ndim << std::endl;
        throw std::runtime_error(strstr.str());
    }

    if (verbose) {

        if (mode == 0) {
            std::cout << "Using Unoptimized Mode" << std::endl;
        } else {
            std::cout << "Using Optimized Mode" << std::endl;
        }

        std::cout << "ha.shape[0]: " << ha.shape[0] << std::endl;
        std::cout << "ha.shape[1]: " << ha.shape[1] << std::endl;
        std::cout << "ha.shape[2]: " << ha.shape[2] << std::endl;
        std::cout << "ha.shape[3]: " << ha.shape[3] << std::endl;

        std::cout << "hb.shape[0]: " << hb.shape[0] << std::endl;
        std::cout << "hb.shape[1]: " << hb.shape[1] << std::endl;
        std::cout << "hb.shape[2]: " << hb.shape[2] << std::endl;
        std::cout << "hb.shape[3]: " << hb.shape[3] << std::endl;

        std::cout << "ha size: " << ha.size * sizeof(lint) << std::endl;
    }

    // reshape hc

    int B, N, M, n;

    B = hc.shape[0];
    N = hc.shape[1];
    M = hc.shape[2];
    n = hc.shape[3];
    lint n_ = n / 2;

    lint *gpu_ptr_a;
    lint *gpu_ptr_b;
    lint *gpu_ptr_c;

    gpuErrchk(cudaMalloc(&gpu_ptr_a, ha.size * sizeof(lint)));
    gpuErrchk(cudaMalloc(&gpu_ptr_b, hb.size * sizeof(lint)));
    gpuErrchk(cudaMalloc(&gpu_ptr_c, hc.size * sizeof(lint)));

    gpuErrchk(cudaMemcpy(gpu_ptr_a, ha.ptr, ha.size * sizeof(lint),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_ptr_b, hb.ptr, hb.size * sizeof(lint),
                         cudaMemcpyHostToDevice));

    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(B, N, M);
    if (mode == 0) {
        batchLongTensorOffsetMult<<<dimGrid, dimBlock>>>(
            gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, B, N, M, n, 0, 0, 0, n_, n_, base);
    } else {
        std::cout << "Not implemented yet" << std::endl;
    }

    lint *ptr = reinterpret_cast<lint *>(hc.ptr);
    gpuErrchk(cudaMemcpy(ptr, gpu_ptr_c, hc.size * sizeof(lint),
                         cudaMemcpyDeviceToHost));

    cudaFree(gpu_ptr_a);
    cudaFree(gpu_ptr_b);
    cudaFree(gpu_ptr_c);
}
