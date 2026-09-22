def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 12
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    dtype = x.dtype
    
    # Initial sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Determine which blocks this rank owns
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Batch ALL 7 rounds into a single all-reduce
    # This reduces collective dispatch overhead
    num_rounds = 7
    mega_batch_size = num_rounds * B * S
    mega_batch = torch.zeros(mega_batch_size, dtype=dtype, device=x.device)
    
    # Pack all 7 rounds into the mega batch
    for round_idx in range(num_rounds):
        buf_offset = round_idx * B * S
        for b in range(B):
            if b in keep:
                mega_batch[buf_offset + b*S : buf_offset + (b+1)*S] = s[b*S:(b+1)*S]
    
    # Single all-reduce for all rounds
    mega_batch_result = xm.all_reduce(xm.REDUCE_SUM, mega_batch)
    
    # Process all rounds locally with the results
    for round_idx in range(num_rounds):
        buf_offset = round_idx * B * S
        acc = torch.zeros_like(s)
        
        if round_idx < 6:
            # Rounds 0-5: apply scaling
            for b in range(B):
                acc[b*S:(b+1)*S] = mega_batch_result[buf_offset + b*S : buf_offset + (b+1)*S] / c[b]
        else:
            # Round 6: no scaling (final round)
            for b in range(B):
                acc[b*S:(b+1)*S] = mega_batch_result[buf_offset + b*S : buf_offset + (b+1)*S]
        
        s = acc
    
    return s