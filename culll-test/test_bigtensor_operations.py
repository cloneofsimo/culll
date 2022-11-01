'''
Tests the following functions

BigTensor.add_gpu
BigTensor.mult_gpu
BigTensor.sub_gpu

BigTensor.get_shift_amount_gpu
BigTensor.shift_gpu
BigTensor.shift_gpu_inplace
'''

from culll import BigTensor
import numpy as np

BASE = 10

def _get_pair_numbers(n = 100):
    B, N, M, n = 2, 2, 2, n

    a = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)

    a = BigTensor(a, BASE)
    b = BigTensor(b, BASE)

    return a, b

def _npvec2int(L, base=10):
    # L = L.tolist()
    return sum([int(base**i) * int(L[i]) for i in range(len(L))])

def get_val(a : BigTensor, i, j, k):
    L = a.at_index(i, j, k)
    if L[-1] >= BASE // 2:

        n_bits = len(L)
        L = _npvec2int(L, BASE)
        return -(BASE**n_bits) + L
    else:
        return _npvec2int(L, BASE)

def test_add():
    # test signed operations
    a, b = _get_pair_numbers(20)
    a, b = a.zero_pad_gpu(22), b.zero_pad_gpu(22) # make sure they are the same size
    c = a.add_gpu(b)
    
    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0) + get_val(b, 0, 0, 0)

def test_mult():
    a, b = _get_pair_numbers(10)
    a, b = a.zero_pad_gpu(12), b.zero_pad_gpu(12) # make sure they are the same size
    #print(a.at_index(0, 0, 0))
    #print(b.at_index(0, 0, 0))
    c = a.mult_gpu(b)
    #print(c.at_index(0, 0, 0))
    
    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0) * get_val(b, 0, 0, 0), f"{get_val(c, 0, 0, 0)} != {get_val(a, 0, 0, 0)} * {get_val(b, 0, 0, 0)}"
