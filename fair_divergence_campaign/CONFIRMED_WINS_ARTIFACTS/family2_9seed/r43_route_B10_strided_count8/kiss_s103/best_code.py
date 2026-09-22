def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 10; W = world_size; L = 3; OFF = 2; STR = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # per-block overlap count c[b] = #ranks whose window covers block b
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # rank-indexed routing: THIS rank masks its own length-L window
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # Create combined mask that includes division (mask * div_factor for each block)
    mask = torch.zeros(B*S, dtype=x.dtype, device=x.device)
    mask_div = torch.zeros(B*S, dtype=x.dtype, device=x.device)
    
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
            div_val = 1.0 / c[b] if c[b] > 0 else 1.0
            mask_div[b*S:(b+1)*S] = div_val
    
    # 6 iterations with division
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask_div)
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask_div)
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask_div)
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask_div)
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask_div)
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask_div)
    
    # Final iteration without division
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask)
    
    return s
