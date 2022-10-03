
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <gpu_utils.h>


using lint = unsigned int;


__global__ void batchLongTensorAdd(
  lint* batched_data_a, lint* batched_data_b, lint* output_data, lint B, lint N, lint M, lint n, lint base = 10)
{
  // input form of B x N x M x n where
  // B is batch size
  // N is number of rows
  // M is number of columns
  // n is number of bits.

  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int col_idx = blockIdx.z * blockDim.z + threadIdx.z;
  // use for loops to iterate over n

  lint sum = 0;
  lint overflow = 0;

  int pos = batch_idx * N * M * n + row_idx * M * n + col_idx * n;

  for (int i = 0; i < n; i++)
  {
    lint a = batched_data_a[pos + i];
    lint b = batched_data_b[pos + i];
    sum = a + b + overflow;
    if (sum >= base)
    {
      overflow = 1;
      sum %= base;
    }
    else
    {
      overflow = 0;
    }
    //
    output_data[pos + i] = sum;
  }
}



void batchLongTensorAddWrapper(
  pybind11::array_t<lint> batched_data_a, pybind11::array_t<lint> batched_data_b, pybind11::array_t<lint> output_data, int base = 10)
{
  pybind11::buffer_info ha = batched_data_a.request();
  pybind11::buffer_info hb = batched_data_b.request();
  pybind11::buffer_info hc = output_data.request();

  if (ha.ndim != 4)
  {
    std::stringstream strstr;
    strstr << "ha.ndim != 4" << std::endl;
    strstr << "ha.ndim: " << ha.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  if (hb.ndim != 4)
  {
    std::stringstream strstr;
    strstr << "hb.ndim != 4" << std::endl;
    strstr << "hb.ndim: " << hb.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  if (hc.ndim != 4)
  {
    std::stringstream strstr;
    strstr << "hc.ndim != 4" << std::endl;
    strstr << "hc.ndim: " << hc.ndim << std::endl;
    throw std::runtime_error(strstr.str());
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

  int B, N, M, n;

  B = ha.shape[0];
  N = ha.shape[1];
  M = ha.shape[2];
  n = ha.shape[3];

  lint* gpu_ptr_a;
  lint* gpu_ptr_b;
  lint* gpu_ptr_c;

  gpuErrchk(cudaMalloc(&gpu_ptr_a, ha.size * sizeof(lint)));
  gpuErrchk(cudaMalloc(&gpu_ptr_b, hb.size * sizeof(lint)));
  gpuErrchk(cudaMalloc(&gpu_ptr_c, hc.size * sizeof(lint)));

  gpuErrchk(cudaMemcpy(gpu_ptr_a, ha.ptr, ha.size * sizeof(lint), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(gpu_ptr_b, hb.ptr, hb.size * sizeof(lint), cudaMemcpyHostToDevice));

  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(B, N, M);

  batchLongTensorAdd << <dimGrid, dimBlock >> > (gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, B, N, M, n, base);
  lint* ptr = reinterpret_cast<lint*>(hc.ptr);
  gpuErrchk(cudaMemcpy(ptr, gpu_ptr_c, hc.size * sizeof(lint), cudaMemcpyDeviceToHost));

  cudaFree(gpu_ptr_a);
  cudaFree(gpu_ptr_b);
  cudaFree(gpu_ptr_c);
}