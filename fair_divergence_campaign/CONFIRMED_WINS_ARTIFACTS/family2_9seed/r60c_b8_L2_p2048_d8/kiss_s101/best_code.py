
def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 8
    L = 2
    OFF = 2
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute division factors as a tensor
    div_factors = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            div_factors[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Precompute which blocks this rank keeps
    start = (rank + OFF) % B
    keep_blocks = [(start + j) % B for j in range(L)]
    
    for iteration in range(7):
        buf = torch.zeros_like(s)
        
        # Copy only this rank's window blocks
        for b in keep_blocks:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by overlap counts with single multiplication
        if iteration < 6:
            s = s * div_factors
    
    return s
