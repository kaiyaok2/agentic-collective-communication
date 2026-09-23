
def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 16
    OFF = 2
    
    # Precompute counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(5):
            c[(st + j) % B] += 1
    
    # Create mask tensor for this rank (reusable)
    start = (rank + OFF) % B
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for j in range(5):
        b = (start + j) % B
        mask[b*S:(b+1)*S] = 1.0
    
    # Create division factor tensor (reusable)
    div_factor = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            div_factor[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterate 7 times
    for iteration in range(7):
        # Apply mask
        buf = s * mask
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by counts using vectorized multiplication
        if iteration < 6:
            s = acc * div_factor
        else:
            s = acc
    
    return s
