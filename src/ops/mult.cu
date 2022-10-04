#include <gpu_utils.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using lint = unsigned int;
using ll = long long uint;

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
__global__ void batchLongTensorMult(lint *batched_data_a,
                                    lint *batched_data_b,
                                    lint *output_data,
                                    lint B, lint N, lint M,
                                    lint n, lint lens,
                                    lint base = 10) {

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;
    // use for loops to iterate over n

    lint sum = 0;
    lint overflow = 0;

    int pos = batch_idx * N * M * n + row_idx * M * n +
              col_idx * n;

    for (int i = 0; i < lens; i++) {
        for (int j = 0; j < lens; j++) {
            lint a = batched_data_a[pos + i];
            lint b = batched_data_b[pos + j];
            sum = a * b + overflow;
            if (sum >= base) {
                overflow = sum / base;
                sum %= base;
            } else {
                overflow = 0;
            }
            //
            output_data[pos + i + j] += sum;
        }
    }
}

void batchLongTensorMultWrapper(
    pybind11::array_t<lint> batched_data_a,
    pybind11::array_t<lint> batched_data_b,
    pybind11::array_t<lint> output_data, int mode = 1,
    int verbose = 0, int base = 10) {
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
            std::cout << "Using Unoptimized Mode"
                      << std::endl;
        } else {
            std::cout << "Using Optimized Mode"
                      << std::endl;
        }

        std::cout << "ha.shape[0]: " << ha.shape[0]
                  << std::endl;
        std::cout << "ha.shape[1]: " << ha.shape[1]
                  << std::endl;
        std::cout << "ha.shape[2]: " << ha.shape[2]
                  << std::endl;
        std::cout << "ha.shape[3]: " << ha.shape[3]
                  << std::endl;

        std::cout << "hb.shape[0]: " << hb.shape[0]
                  << std::endl;
        std::cout << "hb.shape[1]: " << hb.shape[1]
                  << std::endl;
        std::cout << "hb.shape[2]: " << hb.shape[2]
                  << std::endl;
        std::cout << "hb.shape[3]: " << hb.shape[3]
                  << std::endl;

        std::cout << "ha size: " << ha.size * sizeof(lint)
                  << std::endl;
    }

    // reshape hc

    int B, N, M, n;

    B = hc.shape[0];
    N = hc.shape[1];
    M = hc.shape[2];
    n = hc.shape[3];

    lint *gpu_ptr_a;
    lint *gpu_ptr_b;
    lint *gpu_ptr_c;

    gpuErrchk(
        cudaMalloc(&gpu_ptr_a, ha.size * sizeof(lint)));
    gpuErrchk(
        cudaMalloc(&gpu_ptr_b, hb.size * sizeof(lint)));
    gpuErrchk(
        cudaMalloc(&gpu_ptr_c, hc.size * sizeof(lint)));

    gpuErrchk(cudaMemcpy(gpu_ptr_a, ha.ptr,
                         ha.size * sizeof(lint),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_ptr_b, hb.ptr,
                         hb.size * sizeof(lint),
                         cudaMemcpyHostToDevice));

    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(B, N, M);
    if (mode == 0) {
        batchLongTensorMult<<<dimGrid, dimBlock>>>(
            gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, B, N, M, n,
            n / 2 + 2, base);
    } else {
        std::cout << "Not implemented yet" << std::endl;
    }

    lint *ptr = reinterpret_cast<lint *>(hc.ptr);
    gpuErrchk(cudaMemcpy(ptr, gpu_ptr_c,
                         hc.size * sizeof(lint),
                         cudaMemcpyDeviceToHost));

    cudaFree(gpu_ptr_a);
    cudaFree(gpu_ptr_b);
    cudaFree(gpu_ptr_c);
}
