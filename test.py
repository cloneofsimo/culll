from build import culll
import numpy as np
import time


def npvec2int(L, base=10):
    # L = L.tolist()
    return sum([int(base**i) * int(L[i]) for i in range(len(L))])


def cuadd(a, b, base=10):
    B, N, M, _ = a.shape
    nmax = max(a.shape[-1], b.shape[-1]) + 1
    a = np.pad(a, ((0, 0), (0, 0), (0, 0), (0, nmax - a.shape[-1])), mode="constant")
    b = np.pad(b, ((0, 0), (0, 0), (0, 0), (0, nmax - b.shape[-1])), mode="constant")

    c_out = np.zeros((B, N, M, nmax)).astype(np.uint32)

    culll.bignumadd(a, b, c_out, 0, 0, base)
    return c_out


def cumult(a, b, base=10):

    B, N, M, _ = a.shape
    nmax = max(a.shape[-1], b.shape[-1]) * 2
    a = np.pad(a, ((0, 0), (0, 0), (0, 0), (0, nmax - a.shape[-1])), mode="constant")
    b = np.pad(b, ((0, 0), (0, 0), (0, 0), (0, nmax - b.shape[-1])), mode="constant")

    c_out = np.zeros((B, N, M, nmax)).astype(np.uint32)

    culll.bignummult(a, b, c_out, 0, 0, base)
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


def benchmark_mult():

    B, N, M, n = 10, 10, 10, 100

    bef = time.time()

    for _ in range(1000):
        a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
        b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
        c = cumult(a, b, 10)

    print("cuda time: ", time.time() - bef)

    bef = time.time()
    for _ in range(1000):
        for _ in range(B * M * N):
            a = np.random.randint(low=0, high=10, size=(n)).astype(np.uint32)
            b = np.random.randint(low=0, high=10, size=(n)).astype(np.uint32)
            c = npvec2int(a) * npvec2int(b)

    print("numpy time: ", time.time() - bef)


def benchmark_add():

    B, N, M, n = 10, 10, 10, 100

    bef = time.time()

    for _ in range(1000):
        a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
        b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
        c = cuadd(a, b, 10)

    print("cuda time: ", time.time() - bef)

    bef = time.time()
    for _ in range(1000):
        for _ in range(B * M * N):
            a = np.random.randint(low=0, high=10, size=(n)).astype(np.uint32)
            b = np.random.randint(low=0, high=10, size=(n)).astype(np.uint32)
            c = npvec2int(a) + npvec2int(b)

    print("numpy time: ", time.time() - bef)

if __name__ == "__main__":
    # test_add()
    # test_mult()
    #benchmark_mult()
    benchmark_add()
