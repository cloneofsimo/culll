
#include <big_cuops.h>
#include <gpu_utils.h>

__global__ void batchLongTensorNegate(lint *batched_data_a, lint B, lint N,
                                      lint M, lint n, lint base) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int pos = batch_idx * N * M * n + row_idx * M * n + col_idx * n;

    for (int i = 0; i < n; i++) {
        batched_data_a[pos + i] = base - batched_data_a[pos + i];
    }
}

void batchLongTensorNegateWrapper(pybind11::array_t<lint> batched_data_a,
                                  int verbose, int base = 10) {
    pybind11::buffer_info ha = batched_data_a.request();
    int B, N, M, n;

    B = ha.shape[0];
    N = ha.shape[1];
    M = ha.shape[2];
    n = ha.shape[3];

    threedim_checker(ha, "batched_data_a", verbose);

    lint *gpu_ptr_a;
    gpuErrchk(cudaMalloc(&gpu_ptr_a, ha.size * sizeof(lint)));
    gpuErrchk(cudaMemcpy(gpu_ptr_a, ha.ptr, ha.size * sizeof(lint),

    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(B, N, M);

    batchLongTensorNegate<<<dimGrid, dimBlock>>>(gpu_ptr_a, B, N, M, n, base);

    lint *ptr = reinterpret_cast<lint *>(hc.ptr);
    gpuErrchk(cudaMemcpy(ptr, gpu_ptr_c, hc.size * sizeof(lint),
                         cudaMemcpyDeviceToHost));

    cudaFree(gpu_ptr_a);
}