from build import culll
import numpy as np


def npvec2int(L, base=10):
    # L = L.tolist()
    return sum([int(base**i) * L[i] for i in range(len(L))])


def test_add():
    B, N, M, n = 2, 2, 2, 100
    a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    c_out = np.random.randint(low=0, high=1, size=(B, N, M, n)).astype(np.uint32)

    culll.bignumadd(a, b, c_out, 10)

    a = npvec2int(a[0, 0, 0, :])
    b = npvec2int(b[0, 0, 0, :])
    c = npvec2int(c_out[0, 0, 0, :])

    assert a + b == c, f"{a} + {b} != {c}"


def test_mult():
    B, N, M, n = 2, 2, 2, 100
    a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    c_out = np.random.randint(low=0, high=1, size=(B, N, M, n)).astype(np.uint32)

    culll.bignummult(a, b, c_out, 10)

    a = npvec2int(a[0, 0, 0, :])
    b = npvec2int(b[0, 0, 0, :])
    c = npvec2int(c_out[0, 0, 0, :])

    assert a * b == c, f"{a} * {b} != {c}"


if __name__ == "__main__":
    test_add()
    test_mult()
