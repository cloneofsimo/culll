from culll import BigTensor
import numpy as np


def test_signed_overall():
    # test signed operations
    B, N, M, n = 2, 2, 2, 10

    a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    c = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)

    a = BigTensor(a, "gpu", 10)
    b = BigTensor(b, "gpu", 10)
    c = BigTensor(c, "gpu", 10)

    # a.to_device("gpu")
    # b.to_device("gpu")

    d = a._add_gpu(b)

    a.print_slice(0, 1, 0, 1, 0, 2)
    b.print_slice(0, 1, 0, 1, 0, 2)
    d.print_slice(0, 1, 0, 1, 0, 2)

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
