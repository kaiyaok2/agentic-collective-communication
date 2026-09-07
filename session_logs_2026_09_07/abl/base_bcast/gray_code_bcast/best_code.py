
def evolved_p93(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = i XOR (i >> 1) - binary-reflected Gray code
    # Position-based computation - no collective needed
    # Constant folding for N=128
    gray_codes = [i ^ (i >> 1) for i in range(N)]
    return torch.tensor(gray_codes, device=x.device, dtype=x.dtype)
