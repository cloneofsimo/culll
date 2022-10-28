#include <string>
#include <gpu_utils.h>
#include <big_cuops.h>
using lint = unsigned int;



#pragma once
class BigTensor {
public:
    BigTensor() = default;
    BigTensor(lint* data, lint B, lint N, lint M, lint n, lint base) {
        this->data = data;
        this->B = B;
        this->N = N;
        this->M = M;
        this->n = n;
        this->base = base;
        this->device = "cpu";
        gpuErrchk(cudaMalloc(&this->cuda_data, B * N * M * n * sizeof(lint))); // allocate memory on the CPU
    }
    BigTensor(pybind11::array_t<lint> data, std::string device, lint base = 10) {
        pybind11::buffer_info ha = data.request();
        this->data = (lint*)ha.ptr;
        this->B = ha.shape[0];
        this->N = ha.shape[1];
        this->M = ha.shape[2];
        this->n = ha.shape[3];
        this->base = base;
        if (device == "gpu") {
            gpuErrchk(cudaMalloc(&this->cuda_data, this->B * this->N * this->M * this->n * sizeof(lint)));
            gpuErrchk(cudaMemcpy(this->cuda_data, this->data, this->B * this->N * this->M * this->n * sizeof(lint), cudaMemcpyHostToDevice));
            this->device = "gpu";
        }
        else {
            this->device = "cpu";
        }
    }

    lint* data;
    lint* cuda_data;
    lint B;
    lint N;
    lint M;
    lint n;
    lint base;
    std::string device;

    // to device
    void to_device(std::string device) {
        // device either "cpu" or "gpu"
        assert(device == "cpu" || device == "gpu");
        if (device == "gpu") {
            std::cout << "to gpu" << std::endl;

            if (this->device == "gpu") {
                return;
            }
            this->device = "gpu";
            gpuErrchk(cudaMemcpy(this->cuda_data, this->data, this->B * this->N * this->M * this->n * sizeof(lint), cudaMemcpyHostToDevice));
            return;
        }
        else {
            std::cout << "to cpu" << std::endl;
            if (this->device == "cpu") {
                return;
            }
            this->device = "cpu";

            cudaMemcpy(this->data, this->cuda_data, this->B * this->N * this->M * this->n * sizeof(lint), cudaMemcpyDeviceToHost);

            return;
        }
    }

    // Getter
    BigTensor slice(lint a0_s, lint a0_e, lint a1_s, lint a1_e, lint a2_s, lint a2_e, lint a3_s, lint a3_e) {

        this->to_device("cpu"); // TODO: make this more efficient

        lint* data = this->data;
        lint B = this->B;
        lint N = this->N;
        lint M = this->M;
        lint n = this->n;
        lint base = this->base;

        lint B_s = a0_s;
        lint B_e = a0_e;
        lint N_s = a1_s;
        lint N_e = a1_e;
        lint M_s = a2_s;
        lint M_e = a2_e;
        lint n_s = a3_s;
        lint n_e = a3_e;

        lint B_n = B_e - B_s;
        lint N_n = N_e - N_s;
        lint M_n = M_e - M_s;
        lint n_n = n_e - n_s;

        assert(B_n > 0 && N_n > 0 && M_n > 0 && n_n > 0);

        lint* data_slice = new lint[B_n * N_n * M_n * n_n];
        for (lint b = 0; b < B_n; b++) {
            for (lint n = 0; n < N_n; n++) {
                for (lint m = 0; m < M_n; m++) {
                    for (lint i = 0; i < n_n; i++) {
                        data_slice[b * N_n * M_n * n_n + n * M_n * n_n + m * n_n + i] = data[(b + B_s) * N * M * n + (n + N_s) * M * n + (m + M_s) * n + (i + n_s)];
                    }
                }
            }
        }

        return BigTensor(data_slice, B_n, N_n, M_n, n_n, base);
    }

    BigTensor copy() {
        this->to_device("cpu");

        lint B = this->B;
        lint N = this->N;
        lint M = this->M;
        lint n = this->n;
        lint base = this->base;

        lint* data_copy = new lint[B * N * M * n];
        for (lint _b = 0; _b < B; _b++) {
            for (lint _n = 0; _n < N; _n++) {
                for (lint _m = 0; _m < M; _m++) {
                    for (lint _i = 0; _i < n; _i++) {
                        data_copy[_b * N * M * n + _n * M * n + _m * n + _i] = this->data[_b * N * M * n + _n * M * n + _m * n + _i];
                    }
                }
            }
        }

        return BigTensor(data_copy, B, N, M, n, base);
    }

    void write_numpy(pybind11::array_t<lint> data) {
        pybind11::buffer_info ha = data.request();
        lint* data_ptr = (lint*)ha.ptr;
        lint B = ha.shape[0];
        lint N = ha.shape[1];
        lint M = ha.shape[2];
        lint n = ha.shape[3];

        assert(B == this->B && N == this->N && M == this->M && n == this->n);

        this->to_device("cpu");
        for (lint _b = 0; _b < B; _b++) {
            for (lint _n = 0; _n < N; _n++) {
                for (lint _m = 0; _m < M; _m++) {
                    for (lint _i = 0; _i < n; _i++) {
                        data_ptr[_b * N * M * n + _n * M * n + _m * n + _i] = this->data[_b * N * M * n + _n * M * n + _m * n + _i];
                    }
                }
            }
        }
    }

    BigTensor _add_gpu(BigTensor a) {

        BigTensor b = this->copy();
        b.to_device("gpu");

        assert(B == a.B);
        assert(N == a.N);
        assert(M == a.M);
        assert(n == a.n);
        assert(base == a.base);
        assert(device == "gpu");
        assert(a.device == "gpu");


        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(B, N, M);

        batchBigTensorKernelOffsetAdd << <dimGrid, dimBlock >> > (this->cuda_data, a.cuda_data, b.cuda_data, B, N, M, n, 0, 0, 0, n, base);
        return b;
    }

    void print_slice(lint a0_s, lint a0_e, lint a1_s, lint a1_e, lint a2_s, lint a2_e) {
        //assert((this->device == "cpu1", "print_slice only works on cpu"));
        this->to_device("cpu");
        for (lint _b = a0_s; _b < a0_e; _b++) {
            std::cout << "[ ";
            for (lint _n = a1_s; _n < a1_e; _n++) {
                std::cout << "[ ";
                for (lint _m = a2_s; _m < a2_e; _m++) {
                    std::cout << "[ ";
                    for (lint i = 0; i < this->n; i++) {
                        std::cout << this->data[_b * this->N * this->M * this->n + _n * this->M * this->n + _m * this->n + i] << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    }
};
