
def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = popcount(i XOR j)
    # Precompute as 2D list to avoid view operation
    
    values = [[bin(i ^ j).count('1') for j in range(N)] for i in range(N)]
    result = torch.tensor(values, device=x.device, dtype=x.dtype)
    return result
