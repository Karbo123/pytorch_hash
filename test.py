import torch, pytest
from pytorch_hash import pytorch_hash

@pytest.mark.parametrize("num", [10, 64, 1000, 4096, 10000, 65536])
@pytest.mark.parametrize("dim", [2, 4, 6, 8, 10, 64, 128, 512])
@pytest.mark.parametrize("drange", [5, 100, 100000])
@pytest.mark.parametrize("dtype", [torch.half, torch.float32, torch.long])
@pytest.mark.parametrize("device0", ["cpu", "cuda"])
@pytest.mark.parametrize("device1", ["cpu", "cuda"])
def test(num, dim, drange, dtype, device0, device1):
    if dtype in (torch.half, torch.float32): # floating point
        x0 = torch.randn([num, dim], dtype=dtype, device=device0).mul(drange)
    else: # integer
        x0 = torch.randint(-drange, drange, [num, dim], dtype=dtype, device=device0)
    x1 = x0.clone().to(device=device1)
    y0 = pytorch_hash(x0).cpu()
    y1 = pytorch_hash(x1).cpu()
    y_err = (y0 - y1).abs().sum().item() # result must match
    assert y_err == 0
