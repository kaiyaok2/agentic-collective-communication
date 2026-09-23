def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 13
    OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts once (Python-level)
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(4))
        for b in ks:
            c[b] += 1
    
    # Create normalization tensor once (handle division by zero)
    norm = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Create mask once for this rank's kept buckets
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(4))
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # 7 iterations with optimized operations
    for iteration in range(7):
        # Use multiplication instead of zeros_like + slice assignments
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Use multiplication instead of element-wise divisions (except last iteration)
        if iteration < 6:
            s = acc * norm
        else:
            s = acc
    
    return s