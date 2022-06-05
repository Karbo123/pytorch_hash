import os.path as osp
from torch.utils.cpp_extension import load
lib = load("pytorch_hash", osp.join(osp.dirname(osp.abspath(__file__)), "pytorch_hash.cu"))
pytorch_hash = lib.hash
