#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include <big_tensor.h>
#include <gpu_utils.h>

#include "ops/add.cu"
#include "ops/morph.cu"
#include "ops/mult.cu"
#include "ops/shift.cu"

using lint = unsigned int;

namespace py = pybind11;

PYBIND11_MODULE(culll, m) {
    
    py::class_<BigTensor>(m, "BigTensor")
        .def(py::init<pybind11::array_t<lint>, lint>())

        // read-writes
        .def_readwrite("base", &BigTensor::base)
        .def_readwrite("logbase", &BigTensor::logbase)


        // Operations
        .def("add_gpu", &BigTensor::add_gpu)
        .def("mult_gpu", &BigTensor::mult_gpu)
        .def("clz_gpu", &BigTensor::clz_gpu)
        .def("shift_gpu_inplace", &BigTensor::shift_gpu_inplace)

        // Morphs
        .def("redigit_gpu", &BigTensor::redigit_gpu)
        .def("zero_pad_gpu", &BigTensor::zero_pad_gpu)
        .def("negate_gpu", &BigTensor::negate_gpu)
        .def("negate_gpu_inplace", &BigTensor::negate_gpu_inplace)
        .def("as_binary", &BigTensor::as_binary)

        //ios
        .def("copy", &BigTensor::copy)
        .def("print_slice", &BigTensor::print_slice)
        .def("write_numpy", &BigTensor::write_numpy)
        .def("size", &BigTensor::size)
        .def("at_index", &BigTensor::at_index);
}