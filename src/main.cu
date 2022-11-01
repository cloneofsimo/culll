#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
// #include <big_cuops.h>
#include "ops/add.cu"
#include "ops/morph.cu"
#include "ops/mult.cu"
#include <big_tensor.h>
#include <gpu_utils.h>

using lint = unsigned int;

namespace py = pybind11;

PYBIND11_MODULE(culll, m) {
    m.def("badd", batchBigTensorAddWrapper);
    m.def("bmult", batchBigTensorMultWrapper);
    m.def("bnegate", batchBigTensorNegateWrapper);
    m.def("bdigit_resize", batchBigTensorDigitResizeWrapper);

    py::class_<BigTensor>(m, "BigTensor")
        .def(py::init<pybind11::array_t<lint>, lint>())
        .def("copy", &BigTensor::copy)
        .def("print_slice", &BigTensor::print_slice)
        // Operations
        .def("add_gpu", &BigTensor::add_gpu)
        .def("mult_gpu", &BigTensor::mult_gpu)
        .def("get_shift_amount_gpu", &BigTensor::get_shift_amount_gpu)
        .def("shift_gpu_inplace", &BigTensor::shift_gpu_inplace)
        // Morphs
        .def("redigit_gpu", &BigTensor::redigit_gpu)
        .def("zero_pad_gpu", &BigTensor::zero_pad_gpu)
        .def("negate_gpu", &BigTensor::negate_gpu)
        .def("negate_gpu_inplace", &BigTensor::negate_gpu_inplace)
        .def("write_numpy", &BigTensor::write_numpy)
        .def("size", &BigTensor::size)
        .def("at_index", &BigTensor::at_index);
}