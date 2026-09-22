def r43_route_B11_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 11
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap count c[b] = #ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Determine which blocks this rank's window covers
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Batched strategy: batch the 8 iterations into 4 pairs
    # Each pair performs two mask-reduce-scale operations batched together
    # We'll stack two iterations' buffers and dispatch a single all_reduce
    
    for batch_idx in range(4):
        # Each batch processes 2 iterations
        # Create a buffer that holds 2 copies of the data
        batched_buf = torch.zeros(2 * B * S, dtype=x.dtype)
        
        # First iteration in batch
        for b in range(B):
            if b in keep:
                batched_buf[b * S:(b + 1) * S] = s[b * S:(b + 1) * S]
        
        # Second iteration in batch (if not the last batch which has only 1 more)
        if batch_idx < 3:  # batches 0,1,2 have 2 iterations each
            for b in range(B):
                if b in keep:
                    batched_buf[B * S + b * S:B * S + (b + 1) * S] = s[b * S:(b + 1) * S]
        
        # Single all_reduce for the batched buffer
        acc_batched = xm.all_reduce(xm.REDUCE_SUM, batched_buf)
        
        # Process first iteration result
        for b in range(B):
            s[b * S:(b + 1) * S] = acc_batched[b * S:(b + 1) * S] / c[b]
        
        # Process second iteration result (if applicable)
        if batch_idx < 3:
            for b in range(B):
                s[b * S:(b + 1) * S] = acc_batched[B * S + b * S:B * S + (b + 1) * S] / c[b]
    
    # Handle the last (8th) iteration separately
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b * S:(b + 1) * S] = s[b * S:(b + 1) * S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc
    
    return s