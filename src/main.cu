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
  m.def("badd", batchLongTensorAddWrapper);
  m.def("bmult", batchLongTensorMultWrapper);
  m.def("bnegate", batchLongTensorNegateWrapper);
}