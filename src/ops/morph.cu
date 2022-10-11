
#include <big_cuops.h>
#include <gpu_utils.h>

__global__ void batchLongTensorNegate(lint *batched_data_a, lint B, lint N,
                                      lint M, lint n, lint base) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int pos = batch_idx * N * M * n + row_idx * M * n + col_idx * n;
    int overflow = 1;
    for (int i = 0; i < n; i++) {
        batched_data_a[pos + i] = base - 1 - batched_data_a[pos + i] + overflow;
        if (batched_data_a[pos + i] >= base) {
            overflow = 1;
            batched_data_a[pos + i] -= base;
        } else {
            overflow = 0;
        }
    }
}

__global__ void batchLongTensorDigitResize(lint *batched_data_a, lint *output, lint B, lint N, lint M, lint n1, lint n2, lint base){
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.z * blockDim.z + threadIdx.z;

    int pos = batch_idx * N * M * n1 + row_idx * M * n1 + col_idx * n1;
    int pos_out = batch_idx * N * M * n2 + row_idx * M * n2 + col_idx * n2;

    int fills = (batched_data_a[pos + n1 - 1] >= base / 2) ? base -1 : 0;

    for(int i = 0; i < n1 - 1; i++){
        output[pos_out + i] = batched_data_a[pos + i];
    }
    for(int i = n1 - 1; i < n2; i++){
        output[pos_out + i] = fills;
    }
    output[pos_out + n2 - 1] = batched_data_a[pos + n1 - 1];
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
                         cudaMemcpyHostToDevice));

    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(B, N, M);

    batchLongTensorNegate<<<dimGrid, dimBlock>>>(gpu_ptr_a, B, N, M, n, base);

    lint *ptr = reinterpret_cast<lint *>(ha.ptr);
    gpuErrchk(cudaMemcpy(ptr, gpu_ptr_a, ha.size * sizeof(lint),
                         cudaMemcpyDeviceToHost));

    cudaFree(gpu_ptr_a);
}



void batchLongTensorDigitResizeWrapper(pybind11::array_t<lint> batched_data_a, pybind11::array_t<lint> batched_data_b,
                                  int verbose, int base = 10) {
    pybind11::buffer_info ha = batched_data_a.request();
    pybind11::buffer_info hb = batched_data_b.request();


    int B, N, M, n1, n2;

    B = ha.shape[0];
    N = ha.shape[1];
    M = ha.shape[2];
    n1 = ha.shape[3];
    n2 = hb.shape[3];



    assert(batched_data_a.shape(3) <= batched_data_b.shape(3));

    threedim_checker(ha, "batched_data_a", verbose);


    lint *gpu_ptr_a, *gpu_ptr_b;
    gpuErrchk(cudaMalloc(&gpu_ptr_a, ha.size * sizeof(lint)));
    gpuErrchk(cudaMemcpy(gpu_ptr_a, ha.ptr, ha.size * sizeof(lint),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&gpu_ptr_b, hb.size * sizeof(lint)));
    gpuErrchk(cudaMemcpy(gpu_ptr_b, hb.ptr, hb.size * sizeof(lint),
                         cudaMemcpyHostToDevice));

    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(B, N, M);

    batchLongTensorDigitResize<<<dimGrid, dimBlock>>>(gpu_ptr_a, gpu_ptr_b, B, N, M, n1, n2, base);

    lint *ptr = reinterpret_cast<lint *>(hb.ptr);
    gpuErrchk(cudaMemcpy(ptr, gpu_ptr_b, hb.size * sizeof(lint),
                         cudaMemcpyDeviceToHost));

    cudaFree(gpu_ptr_a);
    cudaFree(gpu_ptr_b);

}

