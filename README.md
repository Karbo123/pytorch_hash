# pytorch_hash

a pytorch library for fast hashing using [FNV-1A](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1a_hash). it supports up to 10 data types: Byte, Char, Double, Float, Int, Long, Short, Half, ComplexFloat, ComplexDouble, and supports both cpu and cuda.

**Installation:**
```bash
git clone git@github.com:Karbo123/pytorch_hash.git --depth=1
cd pytorch_hash && pip install -e .
```
**NOTE:** the library uses JIT compilation, so please make sure you have NVCC compiler

**Usage:** tensor must have `shape == (num_batch_size, num_dim)`, where each row is the data to be hashed. the result has `shape == (num_batch_size, )` of type `torch.long`. example:
```python
from pytorch_hash import pytorch_hash
x = torch.randn([1234, 64], dtype=torch.float32, device="cuda")
x_hashed = pytorch_hash(x)
```

