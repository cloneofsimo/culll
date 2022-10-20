
using lint = unsigned int;



#pragma once
class HPTensor {
public:
    HPTensor(int B, int N, int M, int base, int precision, int* exponent, lint* mantissa, lint* sign) {
        this->B = B;
        this->N = N;
        this->M = M;
        this->precision = precision;
        this->base = base;
        this->exponent = exponent;
        this->mantissa = mantissa;
        this->sign = sign;
        this->logbase = (int)log2(base);
    }
    int B, N, M;
    int precision, base;
    // exponent has B, N, M elements of int
    // mantissa has B, N, M, precision elements of int
    //
    int* exponent;
    lint* mantissa, * sign;
    lint logbase = 0;

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
