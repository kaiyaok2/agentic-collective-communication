
def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute accumulated weights for each bucket
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = r % 6
        for j in range(5):
            A[(start + j) % 6] += w
    
    # Precompute which buckets this rank keeps
    start = rank % 6
    keep = set((start + j) % 6 for j in range(5))
    w = 0.5 + 0.02 * rank
    
    # Create mask tensor once
    mask = torch.zeros(6 * S, device=x.device, dtype=x.dtype)
    for b in range(6):
        if b in keep:
            mask[b*S:(b+1)*S] = w
    
    # Create normalization tensor once
    norm = torch.zeros(6 * S, device=x.device, dtype=x.dtype)
    for b in range(6):
        norm[b*S:(b+1)*S] = 1.0 / A[b]
    
    # Do 6 iterations with normalization
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * norm
    
    # Final iteration without normalization
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
