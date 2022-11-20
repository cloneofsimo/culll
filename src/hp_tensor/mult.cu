#include <big_cuops.h>
#include <gpu_utils.h>

#include "hp_tensor.h"

void HPTensor::mult(HPTensor* t1) {
  // sign
  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(B, N, M);

  batchBigTensorKernelOffsetAdd<<<dimGrid, dimBlock>>>(
      this->sign, t1->sign, this->sign, B, N, M, 1, 0, 0, 0, 1, 2);

  // exponent
  // shift the exponent by 1 for both tensors and multiply.
}
