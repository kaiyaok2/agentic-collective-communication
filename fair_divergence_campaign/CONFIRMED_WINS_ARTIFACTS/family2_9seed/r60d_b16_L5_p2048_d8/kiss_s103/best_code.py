def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 16
    OFF = 2
    L = 5
    W = world_size
    
    # All-reduce to sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Calculate count for each bucket
    c = [0] * B
    for r in range(W):
        start = (r + OFF) % B
        for j in range(L):
            b = (start + j) % B
            c[b] += 1
    
    # Use broadcasting for efficient scaling
    c_tensor = torch.tensor(c, device=s.device, dtype=s.dtype).view(B, 1)
    s_reshaped = s.view(B, S)
    result = (s_reshaped * c_tensor).view(-1)
    
    return result