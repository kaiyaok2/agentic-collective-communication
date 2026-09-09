
def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = popcount(i XOR j)
    # Hamming distance between i and j as bit patterns
    # For small fixed N=16, use constant folding
    
    # Compute popcount(i XOR j) for all pairs at trace time
    def popcount(n):
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count
    
    values = [[popcount(i ^ j) for j in range(N)] for i in range(N)]
    
    return torch.tensor(values, device=x.device, dtype=x.dtype)
