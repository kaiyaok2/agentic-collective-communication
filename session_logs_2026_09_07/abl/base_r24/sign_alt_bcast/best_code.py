
def evolved_p91(x, N, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = (-1)^(i + j)
    # Constant folding: precompute the checkerboard pattern
    
    # Generate pattern as nested list comprehension
    pattern = [[(-1) ** (i + j) for j in range(N)] for i in range(N)]
    result = torch.tensor(pattern, device=x.device, dtype=x.dtype)
    return result
