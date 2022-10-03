#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "ops/add.cu"
#include "ops/mult.cu"

using lint = unsigned int;



PYBIND11_MODULE(culll, m)
{
  m.def("bignumadd", batchLongTensorAddWrapper);
  m.def("bignummult", batchLongTensorMultWrapper);
}