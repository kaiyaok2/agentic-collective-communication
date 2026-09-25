
def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute accumulation weights
    A = [0.0] * 5
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = r % 5
        for j in range(3):
            A[(st + j) % 5] += w
    
    # Build inv_A using list comprehension and repeat
    inv_A_blocks = [torch.full((S,), 1.0/A[b], device=x.device, dtype=x.dtype) for b in range(5)]
    inv_A = torch.cat(inv_A_blocks)
    
    # Build weighted_mask
    start = rank % 5
    keep = set((start + j) % 5 for j in range(3))
    w = 0.5 + 0.02 * rank
    
    mask_blocks = [torch.full((S,), w if b in keep else 0.0, device=x.device, dtype=x.dtype) for b in range(5)]
    weighted_mask = torch.cat(mask_blocks)
    
    # Run 7 iterations
    for iteration in range(7):
        buf = s * weighted_mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 6:
            acc = acc * inv_A
        s = acc
    
    return s
