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
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

BASE = 16
NBITS = 100


def _get_numbers(n, positive=True, base=BASE, mag=-1):
    B, N, M, n = 2, 2, 2, n

    a = np.random.randint(low=0, high=base, size=(B, N, M, n)).astype(np.uint32)
    if positive:
        a[:, :, :, -1] = np.random.randint(
            low=0, high=base // 2, size=(B, N, M)
        ).astype(np.uint32)

    if mag > 0:
        start_idx = math.ceil(math.log(mag, base))
        a[:, :, :, :] = 0
        a[:, :, :, :] = np.random.randint(low=0, high=mag, size=(B, N, M, n)).astype(
            np.uint32
        )

    a = BigTensor(a, base)

    return a


def _get_pair_numbers(n: int, positive: bool = False):

    a, b = _get_numbers(n, positive=positive), _get_numbers(n, positive=positive)

    return a, b


def _npvec2int(L, base=NBITS):
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


def uval(a: BigTensor, base=-1):
    if base == -1:
        base = a.base
    L = a.at_index(0, 0, 0)
    return _npvec2int(L, base)


def test_add():
    # test signed operations
    a, b = _get_pair_numbers(NBITS)
    a, b = a.zero_pad_gpu(NBITS + 2), b.zero_pad_gpu(NBITS + 2)
    c = a.add_gpu(b)

    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0) + get_val(b, 0, 0, 0)


def test_sub():
    # test signed operations
    a, b = _get_pair_numbers(NBITS)
    a, b = a.zero_pad_gpu(NBITS + 2), b.zero_pad_gpu(NBITS + 2)
    c = a.sub_gpu(b)

    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0) - get_val(b, 0, 0, 0)


def test_mult():
    a, b = _get_pair_numbers(NBITS)
    a, b = a.zero_pad_gpu(NBITS + 2), b.zero_pad_gpu(NBITS + 2)
    c = a.mult_gpu(b)

    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0) * get_val(
        b, 0, 0, 0
    ), f"{get_val(c, 0, 0, 0)} != {get_val(a, 0, 0, 0)} * {get_val(b, 0, 0, 0)}"


def test_copy():
    a, b = _get_pair_numbers(NBITS)
    c = a.copy()

    assert get_val(c, 0, 0, 0) == get_val(a, 0, 0, 0)


def test_binary():

    a, b = _get_pair_numbers(NBITS, positive=True)
    a_b = a.as_binary()

    assert val(a, BASE) == val(a_b, 2)


def test_clz_gpu():
    a, b = _get_pair_numbers(NBITS, positive=True)

    c, d = a.clz_gpu(), b.clz_gpu()

    a_b = a.as_binary()

    a_list = reversed(a_b.at_index(0, 0, 0))
    shift_amount = val(c)

    for i, bit in enumerate(a_list):
        if bit == 1:
            assert i == shift_amount
            break


def test_right_shift_clz_inplace():

    a, b = _get_pair_numbers(NBITS, positive=False)
    z1 = uval(a)
    a_clz = a.clz_gpu()

    a.shift_gpu_inplace(a_clz)
    z2 = uval(a)

    assert z2 >= (BASE**NBITS) // 2, f"{z2} < {BASE ** NBITS} // 2"
    assert z2 == z1 * (2 ** val(a_clz)), f"{z2} != {z1} * {2 ** val(a_clz)}"


def test_left_shift():

    a, _ = _get_pair_numbers(NBITS, positive=False)

    z1_list = a.as_binary().at_index(0, 0, 0)

    move_amount = _get_numbers(1, positive=True, base=65536, mag=10)
    move_amount.negate_gpu_inplace()
    a.shift_gpu_inplace(move_amount)
    # print(a.as_binary().at_index(0, 0, 0))
    z2_list = a.as_binary().at_index(0, 0, 0)
    shift_amount = val(move_amount)

    if shift_amount == 0:
        assert z2_list == z1_list, f"{z2_list} != {z1_list}"
    else:
        assert (
            z2_list[:shift_amount] == z1_list[-shift_amount:]
        ), f"{z2_list, z1_list}, {shift_amount}"


def test_div_gpu():
    a, b = _get_pair_numbers(NBITS, positive=True)


if __name__ == "__main__":
    test_add()
    test_sub()
    test_mult()
    test_copy()
    test_clz_gpu()
    test_div_gpu()
    test_binary()
    test_right_shift_clz_inplace()
    test_left_shift()
