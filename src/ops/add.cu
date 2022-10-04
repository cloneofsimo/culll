
#include <big_cuops.h>
#include <gpu_utils.h>

__global__ void batchLongTensorOffsetAdd(lint *batched_data_a,
                                         lint *batched_data_b,
                                         lint *output_data, lint B, lint N,
                                         lint M, lint n, lint a_start,
                                         lint b_start, lint out_start,
                                         lint lens, lint base = 10) {

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;
    // use for loops to iterate over n

    lint sum = 0;
    lint overflow = 0;

    int pos = batch_idx * N * M * n + row_idx * M * n + col_idx * n;

    for (int i = 0; i < lens; i++) {
        sum = batched_data_a[pos + i + a_start] +
              batched_data_b[pos + i + b_start] + overflow;
        if (sum >= base) {
            overflow = 1;
            sum %= base;
        } else {
            overflow = 0;
        }
        output_data[pos + i + out_start] = sum;
    }
    // output_data[pos + lens + out_start] = overflow;
}

void batchLongTensorAddWrapper(pybind11::array_t<lint> batched_data_a,
                               pybind11::array_t<lint> batched_data_b,
                               pybind11::array_t<lint> output_data, int mode,
                               int verbose, int base = 10) {

    pybind11::buffer_info ha = batched_data_a.request();
    pybind11::buffer_info hb = batched_data_b.request();
    pybind11::buffer_info hc = output_data.request();

    int B, N, M, n;

    B = ha.shape[0];
    N = ha.shape[1];
    M = ha.shape[2];
    n = ha.shape[3];

    threedim_checker(ha, "batched_data_a", verbose);
    threedim_checker(hb, "batched_data_b", verbose, B, N, M);
    threedim_checker(hc, "output_data", verbose, B, N, M);

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

    batchLongTensorOffsetAdd<<<dimGrid, dimBlock>>>(
        gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, B, N, M, n, 0, 0, 0, n, base);
    lint *ptr = reinterpret_cast<lint *>(hc.ptr);
    gpuErrchk(cudaMemcpy(ptr, gpu_ptr_c, hc.size * sizeof(lint),
                         cudaMemcpyDeviceToHost));

    cudaFree(gpu_ptr_a);
    cudaFree(gpu_ptr_b);
    cudaFree(gpu_ptr_c);
}