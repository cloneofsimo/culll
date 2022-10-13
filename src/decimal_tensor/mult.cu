#include "hp_tensor.h"


HPTensor HPTensor::mult(HPTensor* t1) {
    int B, N, M;
    HPTensor t3(B, N, M, t1->precision, t1->exponent, t1->mantissa, t1->sign);
    
}