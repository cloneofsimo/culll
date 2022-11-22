#include <big_cuops.h>
#include <gpu_utils.h>

#include <string>

#include "big_tensor.h"

using lint = unsigned int;

namespace py = pybind11;

#pragma once
class HPTensor {
 public:
  BigTensor *mantissa, *exponent, *sign;
  lint B, N, M, n, base, logbase;
  HPTensor(py::array_t<lint> mantissa, py::array_t<lint> exponent,
           py::array_t<lint> sign, lint base) {
    this->mantissa = new BigTensor(mantissa, base);
    this->exponent = new BigTensor(exponent, 1 << 16);
    this->sign = new BigTensor(sign, 2);
    this->base = base;
    this->logbase = (lint)round(log2(base));
    this->B = this->mantissa->B;
    this->N = this->mantissa->N;
    this->M = this->mantissa->M;
    this->n = this->mantissa->n;
  }

  HPTensor(BigTensor mantissa, BigTensor exponent, BigTensor sign) {
    this->mantissa = new BigTensor(mantissa);
    this->exponent = new BigTensor(exponent);
    this->sign = new BigTensor(sign);
    this->base = mantissa.base;
    this->logbase = mantissa.logbase;
    this->B = mantissa.B;
    this->N = mantissa.N;
    this->M = mantissa.M;
    this->n = mantissa.n;
  }

  ~HPTensor() {
    delete mantissa;
    delete exponent;
    delete sign;
  }

  HPTensor(const HPTensor& other) {
    this->mantissa = new BigTensor(*other.mantissa);
    this->exponent = new BigTensor(*other.exponent);
    this->sign = new BigTensor(*other.sign);
    this->base = other.base;
    this->logbase = other.logbase;
    this->B = other.B;
    this->N = other.N;
    this->M = other.M;
    this->n = other.n;
  }

  HPTensor& operator=(const HPTensor& other) {
    this->mantissa = other.mantissa;
    this->exponent = other.exponent;
    this->sign = other.sign;
    this->base = other.base;
    this->logbase = other.logbase;
    this->B = other.B;
    this->N = other.N;
    this->M = other.M;
    this->n = other.n;
    return *this;
  }

  HPTensor mult_gpu(HPTensor& other) {
    BigTensor _c_mantissa = *this->mantissa * *other.mantissa;
    BigTensor _c_exponent = *this->exponent + *other.exponent;
    BigTensor _c_sign = *this->sign + *other.sign;
    return HPTensor(_c_mantissa, _c_exponent, _c_sign);
  }

  void normalize() {
    BigTensor mantis_clz = this->mantissa->clz_gpu();
    this->exponent->sub_gpu(mantis_clz);
    this->mantissa->shift_gpu_inplace(mantis_clz);
  }
};
