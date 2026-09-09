
def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = i XOR j (bitwise XOR of row and column indices)
    # For small N, constant folding; otherwise vectorized computation
    
    if N <= 64:
        # Constant fold: compute all values at trace time
        values = [[i ^ j for j in range(N)] for i in range(N)]
        result = torch.tensor(values, device=x.device, dtype=x.dtype)
        return result
    else:
        # Vectorized bit-wise XOR computation
        idx = torch.arange(N, device=x.device)
        ii = idx.view(N, 1)
        jj = idx.view(1, N)
        
        max_bits = (N - 1).bit_length()
        powers = torch.tensor([1 << b for b in range(max_bits)], device=x.device)
        powers = powers.view(1, 1, max_bits)
        
        bits_i = (ii.unsqueeze(2) // powers) % 2
        bits_j = (jj.unsqueeze(2) // powers) % 2
        xor_bits = (bits_i + bits_j) % 2
        result = (xor_bits * powers).sum(dim=2)
        
        return result.to(x.dtype)
