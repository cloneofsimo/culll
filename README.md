# cuLLL

CUDA optimized big-integer operations for lattice reduction.

Currently WIP.

## Features

- CUDA optimized big-integer operations, including

  - 3 dimensional bigint Addition
  - 3 dimensional bigint Multiplication
  - 3 dimensional bigint Subtraction
  - Negative number, arbitrary base complement
  - Reshape, numpy-like operations

- Arbitrary Precision Tensor operations, including
  - Shift Left, Right
  - Copy, Swap

## TODO

- [ ] Karatsuba multiplication
- [ ] Add more tests
- [ ] Add more documentation
- [ ] Carry-lookahead addition
- [ ] Arbitrary precision add, mult, subtract, int conversion, compare, inner product
- [ ] View, Reshape, Transpose
- [ ] Implement stubtests: https://mypy.readthedocs.io/en/stable/stubtest.html
- [ ] Automate stubgen