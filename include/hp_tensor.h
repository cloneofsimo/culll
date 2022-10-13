


#pragma once
class HPTensor{
public:
    HPTensor(int B, int N, int M, int precision, int *exponent, int *mantissa, int *sign){
        this->B = B;
        this->N = N;
        this->M = M;
        this->precision = precision;
        this->exponent = exponent;
        this->mantissa = mantissa;
        this->sign = sign;
    }
    int B, N, M;
    int precision;
    int* exponent, *mantissa, *sign;

    HPTensor astype(std::string type);
    HPTensor add(HPTensor* other);
    HPTensor mult(HPTensor* other);
    HPTensor div(HPTensor* other);
    HPTensor shift(int shift);
};
