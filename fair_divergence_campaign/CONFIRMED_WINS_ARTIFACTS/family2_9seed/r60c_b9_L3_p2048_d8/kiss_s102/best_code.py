def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size; B = 9; OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(3):
            c[(st + j) % B] += 1
    
    # Pre-compute which buckets this rank keeps
    start = (rank + OFF) % B
    keep_set = set((start + j) % B for j in range(3))
    
    # Build mask and norm by computing each bucket once
    mask_parts = []
    norm_parts = []
    for b in range(B):
        m_val = 1.0 if b in keep_set else 0.0
        n_val = 1.0 / c[b] if c[b] > 0 else 1.0
        mask_parts.append(torch.full((S,), m_val, device=x.device, dtype=x.dtype))
        norm_parts.append(torch.full((S,), n_val, device=x.device, dtype=x.dtype))
    
    mask = torch.cat(mask_parts)
    norm = torch.cat(norm_parts)
    
    # 6 rounds with normalization
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * mask) * norm
    
    # Last round without normalization
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask)
    
    return s