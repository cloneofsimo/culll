#include "hp_tensor.h"


HPTensor HPTensor::mult(HPTensor &t1, HPTensor &t2) {
    int B, N, M;
    HPTensor t3(B, N, M, t1.precision, t1.exponent, t1.sign);
    
}