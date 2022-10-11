from build import culll
import numpy as np
import time

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



class BigTensor:

    def __init__(self, data : np.ndarray, base : int = 10):
        self.data = data
        self.base = base
        self.shape = data.shape

    def __add__(self, other):

        B, N, M, n = self.shape
        B, N, M, m = other.shape

        _n = max(n, m)

        b1 = self._resize(_n)
        b2 = other._resize(_n)
        
        c_out = np.zeros((B, N, M, _n)).astype(np.uint32)
        culll.badd(b1, b2, c_out, 0, 0, self.base)
        
        return BigTensor(c_out, self.base)
    
    def __mul__(self, other):
       
        B, N, M, n = self.shape
        B, N, M, m = other.shape

        b1 = self._resize(n + m)
        b2 = other._resize(n + m)
        
        c_out = np.zeros((B, N, M, n + m)).astype(np.uint32)
        culll.bmult(b1, b2, c_out, 0, 0, self.base)
        
        return BigTensor(c_out, self.base)

    def _resize(self, m):

        B, N, M, n = self.shape
        output = np.zeros((B, N, M, m)).astype(np.uint32)
        culll.bdigit_resize(self.data, output, 0, self.base)

        return output

    def __neg__(self):

        B, N, M, n = self.shape
        output = np.zeros((B, N, M, n)).astype(np.uint32)
        culll.bnegate(self.data, output, 0, self.base)

        return BigTensor(output, self.base)

    def at(self,x1, x2, x3):
        return npvec2signedint(self.data[x1, x2, x3, :])



def test_unsigned_overall():
    
    # test unsigned operations

    B, N, M, n = 2, 2, 2, 10
    a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    c = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)

    
    # make it positive
    a[..., -1] = 0
    b[..., -1] = 0
    c[..., 2:] = 0
    
    a = BigTensor(a)
    b = BigTensor(b)
    c = BigTensor(c)

    # check resize operation
    a._resize(12)


    a_at0 = a.at(0, 0, 0)
    b_at0 = b.at(0, 0, 0)
    c_at0 = c.at(0, 0, 0)

    # 
    #print(a * b)

    f1 = lambda x, y: x * y
    f2 = lambda x, y: x * y + x
    f3 = lambda x, y, z : x * x * x + y * y * y + z * z * z

    assert f1(a_at0, b_at0) == f1(a, b).at(0, 0, 0)
    assert f2(a_at0, b_at0) == f2(a, b).at(0, 0, 0)
    assert f3(a_at0, b_at0, c_at0) == f3(a, b, c).at(0, 0, 0)
    

def test_signed_overall():
    # test signed operations
    B, N, M, n = 2, 2, 2, 10

    a = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    b = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)
    c = np.random.randint(low=0, high=10, size=(B, N, M, n)).astype(np.uint32)


    a = BigTensor(a)
    b = BigTensor(b)
    c = BigTensor(c)


    a_at0 = a.at(0, 0, 0)
    b_at0 = b.at(0, 0, 0)
    c_at0 = c.at(0, 0, 0)

    # 

    f1 = lambda x, y: (x + y - x - y - y) * y * y
    f2 = lambda x, y: y * x + x * x - y + x
    f3 = lambda x, y, z : x * x * x - y * y * y - z * z * z

    assert f1(a_at0, b_at0) == f1(a, b).at(0, 0, 0)
    assert f2(a_at0, b_at0) == f2(a, b).at(0, 0, 0)
    assert f3(a_at0, b_at0, c_at0) == f3(a, b, c).at(0, 0, 0)

if __name__ == "__main__":
    
    test_unsigned_overall()