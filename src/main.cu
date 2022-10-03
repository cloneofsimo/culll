#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "ops/add.cu"

using lint = unsigned int;

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  }
}

void batchLongTensorAddWrapper(
  pybind11::array_t<lint> batched_data_a, pybind11::array_t<lint> batched_data_b, pybind11::array_t<lint> output_data, lint B, lint N, lint M, lint n)
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

  batchLongTensorAdd << <dimGrid, dimBlock >> > (gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, B, n, N, M);
  lint* ptr = reinterpret_cast<lint*>(hc.ptr);
  gpuErrchk(cudaMemcpy(ptr, gpu_ptr_c, hc.size * sizeof(lint), cudaMemcpyDeviceToHost));

  cudaFree(gpu_ptr_a);
  cudaFree(gpu_ptr_b);
  cudaFree(gpu_ptr_c);
}

PYBIND11_MODULE(culll, m)
{
  m.def("bignumadd", batchLongTensorAddWrapper);
}