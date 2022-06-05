import os.path as osp
from torch.utils.cpp_extension import load
_lib = load("_lib", osp.join(osp.dirname(osp.abspath(__file__)), "pytorch_hash.cu"))
pytorch_hash = _lib.hash
