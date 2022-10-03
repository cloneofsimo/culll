from build import culll
import numpy as np


def npvec2int(L, base=10):
    # L = L.tolist()
    return sum([int(base**i) * int(L[i]) for i in range(len(L))])


def cuadd(a, b, base=10):
    B, N, M, _ = a.shape
    nmax = max(a.shape[-1], b.shape[-1]) + 1
    a = np.pad(a, ((0, 0), (0, 0), (0, 0), (0, nmax - a.shape[-1])), mode="constant")
    b = np.pad(b, ((0, 0), (0, 0), (0, 0), (0, nmax - b.shape[-1])), mode="constant")

    c_out = np.zeros((B, N, M, nmax)).astype(np.uint32)

    culll.bignumadd(a, b, c_out, base)
    return c_out


def cumult(a, b, base=10):

    B, N, M, _ = a.shape
    nmax = max(a.shape[-1], b.shape[-1]) * 2
    a = np.pad(a, ((0, 0), (0, 0), (0, 0), (0, nmax - a.shape[-1])), mode="constant")
    b = np.pad(b, ((0, 0), (0, 0), (0, 0), (0, nmax - b.shape[-1])), mode="constant")

    c_out = np.zeros((B, N, M, nmax)).astype(np.uint32)

    culll.bignummult(a, b, c_out, 0, 1, base)
    return c_out


def test_add():
    B, N, M, n = 2, 2, 2, 4
    a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    c_out = cuadd(a, b, 10)

    a = npvec2int(a[0, 0, 0, :])
    b = npvec2int(b[0, 0, 0, :])
    c = npvec2int(c_out[0, 0, 0, :])

    assert a + b == c, f"{a} + {b} != {c}"


def test_mult():
    B, N, M, n = 2, 2, 2, 100
    a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)

    c_out = cumult(a, b, 10)

    a = npvec2int(a[0, 0, 0, :])
    b = npvec2int(b[0, 0, 0, :])
    c = npvec2int(c_out[0, 0, 0, :])

    assert a * b == c, f"{a} * {b} != {c}"


if __name__ == "__main__":
    test_add()
    test_mult()
