
def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384; B = 10; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute overlap counts per block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Precompute this rank's window blocks
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # Create normalization tensor (inverse of overlap counts)
    norm_list = []
    for b in range(B):
        factor = 1.0 / c[b] if c[b] > 0 else 1.0
        norm_list.extend([factor] * S)
    norm_tensor = torch.tensor(norm_list, device=x.device, dtype=x.dtype)
    
    # Perform 7 iterations
    for iteration in range(7):
        # Mask: keep only this rank's window blocks
        buf = torch.zeros_like(s)
        for b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # Aggregate
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by overlap count (skip on last iteration)
        if iteration < 6:
            s = s * norm_tensor
    
    return s
