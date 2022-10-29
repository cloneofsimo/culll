#include <big_cuops.h>
#include <gpu_utils.h>
#include <string>
using lint = unsigned int;

#pragma once
class BigTensor {
  public:
    BigTensor(lint *i_data, lint B, lint N, lint M, lint n, lint base) {
        this->data = new lint[B * M * N * n];
        memcpy(this->data, i_data, B * M * N * n * sizeof(lint));

        this->B = B;
        this->N = N;
        this->M = M;
        this->n = n;
        this->base = base;
        gpuErrchk(cudaMalloc(&this->cuda_data, this->B * this->N * this->M *
                                                   this->n * sizeof(lint)));
        gpuErrchk(
            cudaMemcpy(this->cuda_data, this->data,
                       this->B * this->N * this->M * this->n * sizeof(lint),
                       cudaMemcpyHostToDevice));
    }

    BigTensor(lint B, lint N, lint M, lint n, lint base) {
        this->B = B;
        this->N = N;
        this->M = M;
        this->n = n;
        this->base = base;
        this->data = new lint[this->B * this->N * this->M * this->n];
        gpuErrchk(cudaMalloc(&this->cuda_data, this->B * this->N * this->M *
                                                   this->n * sizeof(lint)));
    }

    BigTensor(pybind11::array_t<lint> data, lint base = 10) {
        pybind11::buffer_info ha = data.request();
        this->B = ha.shape[0];
        this->N = ha.shape[1];
        this->M = ha.shape[2];
        this->n = ha.shape[3];
        this->data = new lint[this->B * this->N * this->M * this->n];
        memcpy(this->data, ha.ptr, ha.size * sizeof(lint));

        this->base = base;
        gpuErrchk(cudaMalloc(&this->cuda_data, this->B * this->N * this->M *
                                                   this->n * sizeof(lint)));
        gpuErrchk(
            cudaMemcpy(this->cuda_data, this->data,
                       this->B * this->N * this->M * this->n * sizeof(lint),
                       cudaMemcpyHostToDevice));
    }

    ~BigTensor() {
        gpuErrchk(cudaFree(this->cuda_data));
        delete[] this->data;
    }

    // copy constructor
    BigTensor(const BigTensor &other) {
        this->B = other.B;
        this->N = other.N;
        this->M = other.M;
        this->n = other.n;
        this->base = other.base;
        this->data = new lint[this->B * this->N * this->M * this->n];
        memcpy(this->data, other.data,
               this->B * this->N * this->M * this->n * sizeof(lint));
        gpuErrchk(cudaMalloc(&this->cuda_data, this->B * this->N * this->M *
                                                   this->n * sizeof(lint)));
        gpuErrchk(
            cudaMemcpy(this->cuda_data, other.cuda_data,
                       this->B * this->N * this->M * this->n * sizeof(lint),
                       cudaMemcpyDeviceToDevice));
    }

    // copy assignment operator
    BigTensor &operator=(const BigTensor &other) {
        if (this != &other) {
            this->B = other.B;
            this->N = other.N;
            this->M = other.M;
            this->n = other.n;
            this->base = other.base;
            delete[] this->data;
            this->data = new lint[this->B * this->N * this->M * this->n];
            memcpy(this->data, other.data,
                   this->B * this->N * this->M * this->n * sizeof(lint));
            gpuErrchk(cudaFree(this->cuda_data));
            gpuErrchk(cudaMalloc(&this->cuda_data, this->B * this->N * this->M *
                                                       this->n * sizeof(lint)));
            gpuErrchk(
                cudaMemcpy(this->cuda_data, other.cuda_data,
                           this->B * this->N * this->M * this->n * sizeof(lint),
                           cudaMemcpyDeviceToDevice));
        }
        return *this;
    }

    BigTensor copy() {
        BigTensor copy(*this);
        return copy;
    }

    lint *data;
    lint *cuda_data;
    lint B;
    lint N;
    lint M;
    lint n;
    lint base;

    // to device
    void _sync() {
        cudaMemcpy(this->data, this->cuda_data,
                   this->B * this->N * this->M * this->n * sizeof(lint),
                   cudaMemcpyDeviceToHost);
        return;
    }

    // Getter
    BigTensor slice(lint a0_s, lint a0_e, lint a1_s, lint a1_e, lint a2_s,
                    lint a2_e, lint a3_s, lint a3_e) {

        lint _B = a0_e - a0_s;
        lint _N = a1_e - a1_s;
        lint _M = a2_e - a2_s;
        lint _n = a3_e - a3_s;

        assert(_B > 0 && _N > 0 && _M > 0 && _n > 0);

        lint *_data = new lint[_B * _N * _M * _n];
        _sync();
        for (int i = 0; i < _B; i++) {
            for (int j = 0; j < _N; j++) {
                for (int k = 0; k < _M; k++) {
                    for (int l = 0; l < _n; l++) {
                        _data[i * _N * _M * _n + j * _M * _n + k * _n + l] =
                            this->data[(i + a0_s) * this->N * this->M *
                                           this->n +
                                       (j + a1_s) * this->M * this->n +
                                       (k + a2_s) * this->n + (l + a3_s)];
                    }
                }
            }
        }

        return BigTensor(_data, _B, _N, _M, _n, this->base);
    }

    BigTensor redigit(int n2) {
        BigTensor tmp = BigTensor(this->B, this->N, this->M, n2, this->base);
        _sync();
        for (int i = 0; i < this->B; i++) {
            for (int j = 0; j < this->N; j++) {
                for (int k = 0; k < this->M; k++) {
                    for (int l = 0; l < this->n; l++) {
                        tmp.data[i * this->N * this->M * n2 + j * this->M * n2 +
                                 k * n2 + l] =
                            this->data[i * this->N * this->M * this->n +
                                       j * this->M * this->n + k * this->n + l];
                    }
                }
            }
        }
        return tmp;
    }

    void write_numpy(pybind11::array_t<lint> data) {
        pybind11::buffer_info ha = data.request();
        lint *data_ptr = (lint *)ha.ptr;
        lint B = ha.shape[0];
        lint N = ha.shape[1];
        lint M = ha.shape[2];
        lint n = ha.shape[3];

        assert(B == this->B && N == this->N && M == this->M && n == this->n);
        _sync();
        for (lint _b = 0; _b < B; _b++) {
            for (lint _n = 0; _n < N; _n++) {
                for (lint _m = 0; _m < M; _m++) {
                    for (lint _i = 0; _i < n; _i++) {
                        data_ptr[_b * N * M * n + _n * M * n + _m * n + _i] =
                            this->data[_b * N * M * n + _n * M * n + _m * n +
                                       _i];
                    }
                }
            }
        }
    }

    BigTensor _add_gpu(BigTensor a) {

        BigTensor c = this->copy();

        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(B, N, M);

        batchBigTensorKernelOffsetAdd<<<dimGrid, dimBlock>>>(
            this->cuda_data, a.cuda_data, c.cuda_data, B, N, M, n, 0, 0, 0, n,
            base);

        return c;
    }

    BigTensor _mult_gpu(BigTensor a) {
        BigTensor c = BigTensor(B, N, M, a.n + this->n, base);
        lint a_len = a.n;
        lint b_len = this->n;

        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(B, N, M);

        batchBigTensorKernelOffsetMultShared256<<<dimGrid, dimBlock>>>(
            a.cuda_data, this->cuda_data, c.cuda_data, B, N, M, a_len + b_len,
            0, 0, 0, a_len, b_len, this->base);

        return c;
    }

    void print_slice(lint a0_s, lint a0_e, lint a1_s, lint a1_e, lint a2_s,
                     lint a2_e) {
        _sync();
        for (lint _b = a0_s; _b < a0_e; _b++) {
            std::cout << "[ ";
            for (lint _n = a1_s; _n < a1_e; _n++) {
                std::cout << "[ ";
                for (lint _m = a2_s; _m < a2_e; _m++) {
                    std::cout << "[ ";
                    for (lint i = 0; i < this->n; i++) {
                        std::cout
                            << this->data[_b * this->N * this->M * this->n +
                                          _n * this->M * this->n +
                                          _m * this->n + i]
                            << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "]" << std::endl;
        }
    }
};
