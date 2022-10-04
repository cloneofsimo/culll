#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma once
using lint = unsigned int;

#pragma once
__global__ void batchLongTensorOffsetMult(lint *batched_data_a,
                                          lint *batched_data_b,
                                          lint *output_data, lint B, lint N,
                                          lint M, lint n, lint a_start,
                                          lint b_start, lint out_start,
                                          lint a_n, lint b_n, lint base);
#pragma once
__global__ void batchLongTensorAdd(lint *batched_data_a, lint *batched_data_b,
                                   lint *output_data, lint B, lint N, lint M,
                                   lint n, lint base);