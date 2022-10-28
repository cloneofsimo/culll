#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#pragma once
using lint = unsigned int;

#pragma once
void threedim_checker(pybind11::buffer_info& ha, std::string name = "a",
    int verbose = 0, int dim0 = -1, int dim1 = -1,
    int dim2 = -1) {

    if (ha.ndim != 4) {
        std::stringstream strstr;
        strstr << name << ".ndim != 4" << std::endl;
        strstr << name << ".ndim: " << ha.ndim << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int dim3[3] = { dim0, dim1, dim2 };
    for (int i = 0; i < 3; i++) {
        if (dim3[i] != -1) {
            if (ha.shape[i] != dim3[i]) {
                std::stringstream strstr;
                strstr << name << ".shape[" << i << "] != " << dim3[i]
                    << std::endl;
                strstr << name << ".shape[" << i << "]: " << ha.shape[i]
                    << std::endl;
                throw std::runtime_error(strstr.str());
            }
        }
    }

    if (verbose) {
        std::cout << name << ".shape[0]: " << ha.shape[0] << std::endl;
        std::cout << name << ".shape[1]: " << ha.shape[1] << std::endl;
        std::cout << name << ".shape[2]: " << ha.shape[2] << std::endl;
        std::cout << name << ".shape[3]: " << ha.shape[3] << std::endl;
    }
}

#pragma once
__global__ void batchBigTensorKernelOffsetMult(lint* batched_data_a,
    lint* batched_data_b,
    lint* output_data, lint B, lint N,
    lint M, lint n, lint a_start,
    lint b_start, lint out_start,
    lint a_n, lint b_n, lint base);

#pragma once
__global__ void batchBigTensorKernelOffsetAdd(lint* batched_data_a,
    lint* batched_data_b,
    lint* output_data, lint B, lint N,
    lint M, lint n, lint a_start,
    lint b_start, lint out_start,
    lint lens, lint base);

// TODO: Implement this in the future for fused operation.
// #pragma once
// __global__ void batchBigTensorKernelOffsetSubtract(lint *batched_data_a,
//                                               lint *batched_data_b,
//                                               lint *output_data, lint B, lint
//                                               N, lint M, lint n, lint
//                                               a_start, lint b_start, lint
//                                               out_start, lint lens, lint
//                                               base);

#pragma once
__global__ void batchBigTensorKernelNegate(lint* batched_data_a, lint B, lint N,
    lint M, lint n, lint base);

#pragma once
__global__ void batchBigTensorKernelDigitResize(lint* batched_data_a, lint* output, lint B, lint N, lint M, lint n1, lint n2);

