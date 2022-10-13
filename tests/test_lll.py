from typing import List
from decimal import *
import random


def _inner_prdt(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])


def _sub(v1, v2, const):
    return [v1[i] - const * v2[i] for i in range(len(v1))]


def _add(v1, v2, const):
    return [v1[i] + const * v2[i] for i in range(len(v1))]


def _copy(_B):
    return [[b for b in _B[i]] for i in range(len(_B))]


def gram_schmit_naive(Qin: List[List[Decimal]]) -> List[List[Decimal]]:
    """Gram-Schmidt orthogonalization of a list of vectors"""
    # hard copy
    Q = _copy(Qin)
    d = len(Q)
    n = len(Q[0])

    assert all(len(q) == n for q in Q)
    mus = [[Decimal(0) for _ in range(d)] for _ in range(d)]
    new_basis = []
    for i in range(d):
        v = Q[i]
        for j in range(i):
            muij = _inner_prdt(Q[i], new_basis[j]) / _inner_prdt(
                new_basis[j], new_basis[j]
            )
            v = _sub(v, new_basis[j], muij)
            mus[i][j] = muij

        new_basis.append(v)

    return new_basis, mus


def _lll_iter(Bin, delta=Decimal(0.75)):

    B = _copy(Bin)
    Q, mus = gram_schmit_naive(B)
    # reduce
    d = len(B)
    n = len(B[0])

    for i in range(1, d):
        for j in reversed(range(i)):
            cij = _inner_prdt(B[i], Q[j]) / _inner_prdt(Q[j], Q[j])

            cij = round(cij)
            # print(f"cij = {cij}")
            B[i] = _sub(B[i], B[j], cij)

    # swap
    for i in range(d - 1):
        v2 = _add(Q[i + 1], Q[i], mus[i + 1][i])
        if delta * _inner_prdt(Q[i], Q[i]) > _inner_prdt(v2, v2):
            B[i], B[i + 1] = B[i + 1], B[i]
            return B, False

    return B, True


def lll_naive(B):

    d = len(B)
    n = len(B[0])

    assert all(len(b) == n for b in B)

    B, isend = _lll_iter(B)

    while not isend:
        B, isend = _lll_iter(B)

    return B


def test_orth():

    n = 100
    d = 99

    Q = [[Decimal(random.randrange(0, 100)) for _ in range(n)] for _ in range(d)]
    Q, _ = gram_schmit_naive(Q)

    for i in range(d):
        for j in range(i):
            assert (
                abs(_inner_prdt(Q[i], Q[j])) < 1e-50
            ), f"i = {i}, j = {j}, inner product = {_inner_prdt(Q[i], Q[j])}"


def generate_bignum(n):
    # generate a bignum
    bignum = 0
    for i in range(n):
        randnum = int(random.random() * pow(2, 32))
        bignum += randnum * 2 ** (32 * i)

    return bignum


def is_lll_reduced(delta: Decimal, B: List[List[Decimal]]) -> bool:
    d = len(B)
    n = len(B[0])

    assert all(len(b) == n for b in B)

    Q, mus = gram_schmit_naive(B)

    for i in range(d):
        for j in range(i):
            assert abs(mus[i][j]) <= 1 / 2

    for i in range(d - 1):
        assert (delta - mus[i + 1][i] * mus[i + 1][i]) * _inner_prdt(
            B[i], B[i]
        ) - _inner_prdt(B[i + 1], B[i + 1]) <= 0

    return True


def test_lll():

    n = 20
    d = 19

    getcontext().prec = 1000

    B = [[Decimal(generate_bignum(5)) for _ in range(n)] for _ in range(d)]

    # B's vector max norm
    max_norm = max([_inner_prdt(b, b) for b in B])
    print(f"max norm = {max_norm:e}")
    B = lll_naive(B)

    assert is_lll_reduced(Decimal(0.75), B)


if __name__ == "__main__":
    # test_orth()

    test_lll()
