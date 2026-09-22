def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    
    # Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap counts (how many ranks' windows cover each block)
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Determine which blocks this rank keeps (its window)
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Phase 1: Aggregate masks 0-3 (first 4 dependent steps)
    # Create a 4x larger buffer to hold all 4 intermediate results
    phase1_buf = torch.zeros(4 * B * S, dtype=x.dtype)
    
    for mask_idx in range(4):
        # For each mask iteration, apply the mask and store in its slice
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        # Store in the corresponding slice of phase1_buf
        phase1_buf[mask_idx * B * S : (mask_idx + 1) * B * S] = buf
    
    # All-reduce the concatenated buffer
    phase1_result = xm.all_reduce(xm.REDUCE_SUM, phase1_buf)
    
    # Apply the 4 mask operations sequentially using the aggregated result
    for mask_idx in range(4):
        acc = phase1_result[mask_idx * B * S : (mask_idx + 1) * B * S].clone()
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    # Phase 2: Aggregate masks 4-6 (next 3 dependent steps)
    # Create a 3x larger buffer
    phase2_buf = torch.zeros(3 * B * S, dtype=x.dtype)
    
    for mask_idx in range(3):
        # For each mask iteration, apply the mask and store in its slice
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        # Store in the corresponding slice of phase2_buf
        phase2_buf[mask_idx * B * S : (mask_idx + 1) * B * S] = buf
    
    # All-reduce the concatenated buffer
    phase2_result = xm.all_reduce(xm.REDUCE_SUM, phase2_buf)
    
    # Apply the 3 mask operations sequentially using the aggregated result
    for mask_idx in range(3):
        acc = phase2_result[mask_idx * B * S : (mask_idx + 1) * B * S].clone()
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    # Final mask operation (mask 7) - just one more all_reduce
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s