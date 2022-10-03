from build import culll
import numpy as np


B, N, M, n = 2, 2, 2, 100
a = np.random.randint(low=0, high=2**32, size=(B, N, M, n)).astype(np.uint32)
b = np.random.randint(low=0, high=2**32, size=(B, N, M, n)).astype(np.uint32)
c = np.random.randint(low=0, high=1, size=(B, N, M, n)).astype(np.uint32)

culll.bignumadd(a, b, c, B, N, M, n)

print(a[0, 0, 0, :], b[0, 0, 0, :], c[0, 0, 0, :])
