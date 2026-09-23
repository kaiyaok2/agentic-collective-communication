def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 10
    OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts once
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Precompute keep set for this rank
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Create combined mask-and-divide tensor for iterations 1-6
    mask_div = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if b in keep and c[b] > 0:
            mask_div[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Create mask-only tensor for final iteration
    mask_final = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if b in keep:
            mask_final[b*S:(b+1)*S] = 1.0
    
    # 6 iterations with division (single multiply-reduce-multiply sequence)
    for _ in range(6):
        buf = s * mask_div
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final iteration without division
    buf = s * mask_final
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
