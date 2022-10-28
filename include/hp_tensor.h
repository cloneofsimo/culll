#include <string>
using lint = unsigned int;



#pragma once
class HPTensor {
public:
    /// @brief Datas are expressed as (1 + a_1 /b + a_2 / b^2  + ....  ) * 2^exp
    /// @param B : the number of batches 
    /// @param N : dim 1
    /// @param M : dim 2
    /// @param base : Mantissa are described in base. Base needs to be power of 2.
    /// @param precision : Number of powers to describe Mantissa.
    /// @param exponent : Exponent. Size : int x B x N x M 
    /// @param mantissa : a_n-1, a_n-2, ... a_1.  : Size : B x N x M x precision
    /// @param sign : Sign of the number. 0 for positive, 1 for negative. : Size : B x N x M
    HPTensor(int B, int N, int M, int base, int precision, int* exponent, lint* mantissa, lint* sign, std::string device = "cpu") {
        this->B = B;
        this->N = N;
        this->M = M;
        this->precision = precision;
        this->base = base;
        this->exponent = exponent;
        this->mantissa = mantissa;
        this->sign = sign;
        this->logbase = (int)log2(base);
        this->device = device;
    }
    int B, N, M;
    int precision, base;
    int* exponent;
    lint* mantissa, * sign;
    lint logbase = 0;
    std::string device;

    HPTensor astype(std::string type);
    void reprecision(int precision);
    void add(HPTensor* other);
    void mult(HPTensor* other);
    void div(HPTensor* other);
    void shift(int shift);
    void reposition(int* amount);
    void swap_exp_maximum(HPTensor* other);
    HPTensor inner_product(HPTensor* other, int idx1, int idx2);
    HPTensor copy();


};
