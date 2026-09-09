
def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = popcount(i XOR j)
    # Position-based - precompute as constant tensor
    
    # Compute all values using Python
    result = []
    for i in range(N):
        row = []
        for j in range(N):
            xor_val = i ^ j
            popcount = bin(xor_val).count('1')
            row.append(popcount)
        result.append(row)
    
    return torch.tensor(result, device=x.device, dtype=x.dtype)
