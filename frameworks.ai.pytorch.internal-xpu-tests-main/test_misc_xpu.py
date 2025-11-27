import pytest
import torch

@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_out_warning(device):
    dtype = torch.float32
    input_tensor = torch.randn(3, 3, dtype=dtype, device=device)
    out = torch.randn(3, 4, dtype=dtype, device=device)  # Wrong shape
    
    # Should warn when resizing non-empty tensor with wrong shape
    with pytest.warns(UserWarning, match="An output with one or more elements"):
        torch.logcumsumexp(input_tensor, 0, out=out)
