
def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    B = 5
    
    # Compute counts
    c = [0] * B
    for r in range(W):
        st = r % B
        ks = set((st + j) % B for j in range(2))
        for b in ks:
            c[b] += 1
    
    # Create normalization tensor once
    norm = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Determine which buckets this rank keeps
    start = rank % B
    keep = set((start + j) % B for j in range(2))
    
    # Initial all reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 rounds - first 6 with normalization, last without
    for round_idx in range(7):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if round_idx < 6:
            acc = acc * norm
        
        s = acc
    
    return s
