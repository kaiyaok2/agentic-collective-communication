def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 10
    W = world_size
    L = 3
    OFF = 2
    STR = 2
    
    # Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Process all 8 iterations in a single batch
    # Concatenate all 8 masked buffers
    all_iterations_buf = torch.zeros(8 * B * S, dtype=x.dtype)
    for iter_idx in range(8):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        all_iterations_buf[iter_idx * B * S:(iter_idx + 1) * B * S] = buf
    
    # Single all_reduce for all iterations
    all_iterations_acc = xm.all_reduce(xm.REDUCE_SUM, all_iterations_buf)
    
    # Unpack and apply scaling for iterations 1-7
    for iter_idx in range(7):
        buf_result = all_iterations_acc[iter_idx * B * S:(iter_idx + 1) * B * S]
        for b in range(B):
            s[b*S:(b+1)*S] = buf_result[b*S:(b+1)*S] / c[b]
    
    # Final iteration (8th): just unpack without scaling
    buf_result = all_iterations_acc[7 * B * S:8 * B * S]
    for b in range(B):
        s[b*S:(b+1)*S] = buf_result[b*S:(b+1)*S]
    
    return s