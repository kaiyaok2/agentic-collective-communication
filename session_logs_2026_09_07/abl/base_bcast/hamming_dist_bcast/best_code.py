
def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = popcount(i XOR j)
    # Hamming distance between i and j interpreted as N-bit integers
    # Constant-fold for N=16 to eliminate runtime arithmetic
    
    # Precompute the entire matrix as a constant
    values = []
    for i in range(N):
        row = []
        for j in range(N):
            xor_val = i ^ j
            popcount = bin(xor_val).count('1')
            row.append(popcount)
        values.extend(row)
    
    result = torch.tensor(values, device=x.device, dtype=x.dtype).view(N, N)
    return result
