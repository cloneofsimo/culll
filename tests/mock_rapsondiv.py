import math
from random import random
from decimal import *


def fastdiv(n, d):
    # largest power of 2 that is not greater than d

    for _logd in range(4000):
        if d < pow(2, _logd):
            break

    dp = Decimal(d) / Decimal(pow(2, (_logd) + 1))
    np = Decimal(n) / Decimal(pow(2, (_logd) + 1))

    x = (48 - 32 * dp) / 17

    for _ in range(10):
        x = x + x * (1 - dp * x)

    return np * x


def generate_bignum(n):
    # generate a bignum
    bignum = 0
    for i in range(n):
        randnum = random() * pow(2, 32)
        randnum = int(randnum)

        bignum += randnum * 2 ** (32 * i)

    return bignum


def test_fastdiv():

    for i in range(10000):

        # create bignum
        getcontext().prec = 1000

        a = generate_bignum(10)
        b = generate_bignum(10)
        a = int(a)
        b = int(b)
        n = a * b
        d = a

        print(f"n: {n:e}, d: {d:e}")

        # compute fastdiv
        fastdiv_res = fastdiv(n, d)

        print(f"diff: {fastdiv_res - b}")
        diff = fastdiv_res - b
        # print(f"{float(diff):e}")
        # check if same upto 100 decimal places

        assert abs(Decimal(diff)) < Decimal(1e-100)


test_fastdiv()
