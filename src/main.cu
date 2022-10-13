#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <big_cuops.h>
#include <gpu_utils.h>
#include "ops/add.cu"
#include "ops/morph.cu"
#include "ops/mult.cu"

using lint = unsigned int;



PYBIND11_MODULE(culll, m)
{
  m.def("badd", batchBigTensorAddWrapper);
  m.def("bmult", batchBigTensorMultWrapper);
  m.def("bnegate", batchBigTensorNegateWrapper);
  m.def("bdigit_resize", batchBigTensorDigitResizeWrapper);
}