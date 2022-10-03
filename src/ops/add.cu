
#include <cuda_runtime.h>

using lint = unsigned int;


__global__ void batchLongTensorAdd(
  lint* batched_data_a, lint* batched_data_b, lint* output_data, lint B, lint N, lint M, lint n)
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
    if (sum <= a)
    {
      overflow = 1;
    }
    else
    {
      overflow = 0;
    }
    output_data[pos + i] = sum;
  }
}