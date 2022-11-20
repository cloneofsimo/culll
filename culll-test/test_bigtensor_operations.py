"""
Tests the following functions

BigTensor.add_gpu
BigTensor.mult_gpu
BigTensor.sub_gpu

BigTensor.clz_gpu
BigTensor.shift_gpu
BigTensor.shift_gpu_inplace
"""

from culll import BigTensor
import numpy as np

BASE = 16


def _get_numbers(n, positive=True):
    B, N, M, n = 2, 2, 2, n

    a = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)
    if positive:
        a[:, :, :, -1] = np.random.randint(
            low=0, high=BASE // 2, size=(B, N, M)
        ).astype(np.uint32)

    a = BigTensor(a, BASE)

    return a


def _get_pair_numbers(n=100, positive=False):

    a, b = _get_numbers(n, positive=positive), _get_numbers(n, positive=positive)

    return a, b


def _npvec2int(L, base=10):
    # L = L.tolist()
    return sum([int(base**i) * int(L[i]) for i in range(len(L))])


def get_val(a: BigTensor, i, j, k, base=-1):
    if base == -1:
        base = a.base

    L = a.at_index(i, j, k)
    if L[-1] >= base // 2:
        n_digits = len(L)
        L = _npvec2int(L, base)
        return -(base**n_digits) + L
    else:
        return _npvec2int(L, base)


def val(a: BigTensor, base=-1):
    return get_val(a, 0, 0, 0, base)


def test_add():
    # test signed operations
    a, b = _get_pair_numbers(20)
    a, b = a.zero_pad_gpu(22), b.zero_pad_gpu(22)  # make sure they are the same size
    c = a.add_gpu(b)

    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0) + get_val(b, 0, 0, 0)


def test_mult():
    a, b = _get_pair_numbers(10)
    a, b = a.zero_pad_gpu(12), b.zero_pad_gpu(12)  # make sure they are the same size
    # print(a.at_index(0, 0, 0))
    # print(b.at_index(0, 0, 0))
    c = a.mult_gpu(b)
    # print(c.at_index(0, 0, 0))

    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0) * get_val(
        b, 0, 0, 0
    ), f"{get_val(c, 0, 0, 0)} != {get_val(a, 0, 0, 0)} * {get_val(b, 0, 0, 0)}"


def test_copy():
    a, b = _get_pair_numbers(10)
    c = a.copy()

    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0)


def test_binary():

    a, b = _get_pair_numbers(10, positive=True)
    a_b = a.as_binary()

    assert val(a, BASE) == val(a_b, 2)


def test_clz_gpu():
    a, b = _get_pair_numbers(10, positive=True)

    c, d = a.clz_gpu(), b.clz_gpu()

    a_b = a.as_binary()

    a_list = reversed(a_b.at_index(0, 0, 0))
    shift_amount = val(c)

    for i, bit in enumerate(a_list):
        if bit == 1:
            assert i == shift_amount
            break


def test_shift_gpu_inplace():

    a, b = _get_pair_numbers(10, positive=True)
    print(a.as_binary().at_index(0, 0, 0))
    a_clz = a.clz_gpu()
    print(
        a.size(),
        a_clz.size(),
    )
    print(a.logbase, a.base)
    print(a_clz.base, a_clz.logbase)
    print(val(a_clz))

    a.shift_gpu_inplace(a_clz)

    print(a.as_binary().at_index(0, 0, 0))


def test_div_gpu():
    a, b = _get_pair_numbers(10, positive=True)


if __name__ == "__main__":
    test_add()
    test_mult()
    test_copy()
    test_clz_gpu()
    test_div_gpu()
    test_binary()
    test_shift_gpu_inplace()
