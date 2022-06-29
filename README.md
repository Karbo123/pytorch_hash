# pytorch_hash

a pytorch library for fast hashing using [FNV-1A](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1a_hash). it supports up to 10 data types: Byte, Char, Double, Float, Int, Long, Short, Half, ComplexFloat, ComplexDouble, and supports both cpu and cuda.

**Installation:**
```bash
pip install git+https://github.com/Karbo123/pytorch_hash.git
```
**NOTE:** the library uses JIT compilation, so please make sure you have NVCC compiler

**Usage:** tensor must have `shape == (num_batch_size, num_dim)`, where each row is the data to be hashed. the result has `shape == (num_batch_size, )` of type `torch.long`. example:
```python
import torch
from pytorch_hash import pytorch_hash
x = torch.randn([1234, 64], dtype=torch.float32, device="cuda")
x_hashed = pytorch_hash(x) # maybe positive or negative numbers
```

you can also compile it using cmake like this:
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/torch/share/cmake/Torch
make
```

**Testing:**
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 pytest -s test.py -p no:warnings -p no:cacheprovider
```

**Related Problem:**
- [time delay of print](https://discuss.pytorch.org/t/a-simple-print-function-could-hugely-can-reduce-the-significant-time-delay/153344/7?u=jiabao)
