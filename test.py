import torch, pytest
from pytorch_hash import pytorch_hash

@pytest.mark.parametrize("num", [10, 64, 1000, 4096, 10000, 65536])
@pytest.mark.parametrize("dim", [2, 4, 6, 8, 10, 64, 128, 512])
@pytest.mark.parametrize("dtype", [torch.half, torch.float32, torch.long])
@pytest.mark.parametrize("device0", ["cpu", "cuda"])
@pytest.mark.parametrize("device1", ["cpu", "cuda"])
def test(num, dim, dtype, device0, device1):
    x0 = torch.randn([num, dim], dtype=dtype, device=device0)
    x1 = x1.clone().to(device=device1)
    y0 = pytorch_hash(x0).cpu()
    y1 = pytorch_hash(x1).cpu()
    y_err = (y0 - y1).abs().sum().item()
    assert y_err == 0, f"error is not zero for (num, dim, dtype, device0, device1) = " \
                       f"{num, dim, dtype, device0, device1}"
