"""Quick test: @nki.jit on real Trainium hardware."""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import torch
import torch_xla as xla


# Test 1: return output from kernel
@nki.jit
def add_one(input_hbm):
    out = nl.ndarray(input_hbm.shape, dtype=input_hbm.dtype, buffer=nl.shared_hbm)
    data = nl.load(input_hbm[0:128, 0:512])
    nl.store(out[0:128, 0:512], data + 1.0)
    return out


device = xla.device()
x = torch.randn(128, 512, device=device, dtype=torch.float32)
y = add_one(x)
xla.step()
expected = x.cpu() + 1.0
print(f"Test 1 (return output): Match={torch.allclose(expected, y.cpu(), atol=1e-5)}")
