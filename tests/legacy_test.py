from build import culll
import numpy as np
import time

BASE = 16


def npvec2int(L, base=10):
    # L = L.tolist()
    return sum([int(base**i) * int(L[i]) for i in range(len(L))])


def npvec2signedint(L, base=10):
    if L[-1] >= base // 2:

        n_bits = len(L)
        L = npvec2int(L)
        return -(base**n_bits) + L
    else:
        return npvec2int(L, base)


def cuadd(a, b, base=10):
    B, N, M, _ = a.shape
    nmax = max(a.shape[-1], b.shape[-1]) + 1
    a = np.pad(a, ((0, 0), (0, 0), (0, 0), (0, nmax - a.shape[-1])), mode="constant")
    b = np.pad(b, ((0, 0), (0, 0), (0, 0), (0, nmax - b.shape[-1])), mode="constant")

    c_out = np.zeros((B, N, M, nmax)).astype(np.uint32)

    culll.badd(a, b, c_out, 0, 0, base)
    return c_out


def cumult(a, b, base=10):

    B, N, M, _ = a.shape
    nmax = max(a.shape[-1], b.shape[-1]) * 2
    a = np.pad(a, ((0, 0), (0, 0), (0, 0), (0, nmax - a.shape[-1])), mode="constant")
    b = np.pad(b, ((0, 0), (0, 0), (0, 0), (0, nmax - b.shape[-1])), mode="constant")

    c_out = np.zeros((B, N, M, nmax)).astype(np.uint32)

    culll.bmult(a, b, c_out, 1, 0, base)
    return c_out


def cusubtract(a, b, base=10):
    np.random.seed(0)
    B, N, M, n = a.shape

    b_ = b.copy()
    culll.bnegate(b_, 0, 10)
    c_out = np.zeros((B, N, M, n)).astype(np.uint32)
    culll.badd(a, b_, c_out, 0, 0, base)

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
    print("Add : Test passed")


def test_mult():
    B, N, M, n = 2, 2, 2, 80

    a = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)

    c_out = cumult(a, b, BASE)

    a = npvec2int(a[0, 0, 0, :])
    b = npvec2int(b[0, 0, 0, :])
    c = npvec2int(c_out[0, 0, 0, :])

    assert a * b == c, f"{a} * {b} != {c}"

    print("Mult : Test passed")


def test_sub():
    # np.random.seed(0)
    B, N, M, n = 1, 1, 1, 5
    a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    a[..., -1] = 1

    c_out = cusubtract(a, b, 10)

    a = npvec2signedint(a[0, 0, 0, :])
    b = npvec2signedint(b[0, 0, 0, :])
    c = npvec2signedint(c_out[0, 0, 0, :])

    assert a - b == c, f"{a} - {b} != {c}, Supposed to be {a - b}"
    print("Subtract & Negation : Test passed")


def benchmark_mult(X=40):

    print("Benchmarking multiplication")

    B, N, M, n = X, X, X, 100

    bef = time.time()

    for _ in range(10):
        a = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)
        b = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)
        c = cumult(a, b, BASE)

    print("cuda time: ", time.time() - bef)

    bef = time.time()
    for _ in range(10):
        for _ in range(B * M * N):
            a = np.random.randint(low=0, high=BASE, size=(n)).astype(np.uint32)
            b = np.random.randint(low=0, high=BASE, size=(n)).astype(np.uint32)
            c = npvec2int(a, BASE) * npvec2int(b, BASE)

    print("numpy time: ", time.time() - bef)


def benchmark_add(X=3):

    print("Benchmarking addition")

    B, N, M, n = X, X, X, 100

    bef = time.time()

    for _ in range(10):
        a = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)
        b = np.random.randint(low=0, high=BASE, size=(B, N, M, n)).astype(np.uint32)
        c = cuadd(a, b, BASE)

    print("cuda time: ", time.time() - bef)

    bef = time.time()
    for _ in range(10):
        for _ in range(B * M * N):
            a = np.random.randint(low=0, high=BASE, size=(n)).astype(np.uint32)
            b = np.random.randint(low=0, high=BASE, size=(n)).astype(np.uint32)
            c = npvec2int(a, BASE) + npvec2int(b, BASE)

    print("numpy time: ", time.time() - bef)


if __name__ == "__main__":
    # test_add()
    test_mult()
    # test_sub()
    benchmark_mult()
    benchmark_add()
