
def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    
    # Compute counts for each bucket
    c = [0] * 10
    for r in range(world_size):
        st = (r + 2) % 10
        ks = set((st + 1*j) % 10 for j in range(4))
        for b in ks:
            c[b] += 1
    
    # Create a mask for which buckets this rank keeps
    start = (rank + 2) % 10
    keep = set((start + 1*j) % 10 for j in range(4))
    
    # Create mask tensor once
    mask = torch.zeros(10 * S, device=x.device, dtype=x.dtype)
    for b in range(10):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Create normalization tensor once (avoid division by zero)
    norm = torch.ones(10 * S, device=x.device, dtype=x.dtype)
    for b in range(10):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with normalization
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * norm
    
    # Final iteration without normalization
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
