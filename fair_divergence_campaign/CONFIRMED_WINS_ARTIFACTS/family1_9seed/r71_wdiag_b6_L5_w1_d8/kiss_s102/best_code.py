
def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    A = [0.0]*6
    for r in range(W):
        w = 0.5 + 0.02*r
        st = (r + 0) % 6
        for j in range(5):
            A[(st + 1*j) % 6] += w
    
    start = (rank + 0) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.5 + 0.02*rank
    
    # Build tensors using torch.cat
    weighted_mask_parts = []
    norm_inv_parts = []
    for b in range(6):
        wm = w if b in keep else 0.0
        ni = 1.0 / A[b]
        weighted_mask_parts.append(torch.full((S,), wm, device=x.device, dtype=x.dtype))
        norm_inv_parts.append(torch.full((S,), ni, device=x.device, dtype=x.dtype))
    
    weighted_mask = torch.cat(weighted_mask_parts)
    norm_inv = torch.cat(norm_inv_parts)
    
    # 6 iterations with normalization
    for _ in range(6):
        buf = s * weighted_mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * norm_inv
    
    # 7th iteration without normalization
    buf = s * weighted_mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
