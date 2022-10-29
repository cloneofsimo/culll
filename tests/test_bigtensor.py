from culll import BigTensor
import numpy as np

BASE = 16

def _get_pair_numbers():
    B, N, M, n = 2, 2, 2, BASE

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
    if L[-1] >= a.base // 2:

        n_bits = len(L)
        L = _npvec2int(L)
        return -(a.base**n_bits) + L
    else:
        return _npvec2int(L, a.base)

def at_0(a : BigTensor):
    return a.at(0, 0, 0)

def test_signed_overall():
    # test signed operations
    a, b = _get_pair_numbers()
    d = a._add_gpu(b)

    assert get_val(d, 0, 0, 0) == get_val(a, 0, 0, 0) + get_val(b, 0, 0, 0)

    print(a.size())
    print(at_0(d))
    """
    a_at0 = a.at(0, 0, 0)
    b_at0 = b.at(0, 0, 0)
    c_at0 = c.at(0, 0, 0)
    print(a_at0, b_at0, c_at0)

    a = a._resize(20)
    b = b._resize(20)
    c = c._resize(20)

    a_at0 = a.at(0, 0, 0)
    b_at0 = b.at(0, 0, 0)
    c_at0 = c.at(0, 0, 0)

    print(a_at0, b_at0, c_at0)

    f1 = lambda x, y: x + y - x - y + x + x
    f2 = lambda x, y: y * x + x * x - y + x
    f3 = lambda x, y, z: x * x * x - y * y * y - z * z * z

    assert f1(a_at0, b_at0) == f1(a, b).at(0, 0, 0), print(
        f"{f1(a_at0, b_at0)} != {f1(a, b).at(0, 0, 0)}"
    )
    assert f2(a_at0, b_at0) == f2(a, b).at(0, 0, 0)
    assert f3(a_at0, b_at0, c_at0) == f3(a, b, c).at(0, 0, 0)
    """


test_signed_overall()
