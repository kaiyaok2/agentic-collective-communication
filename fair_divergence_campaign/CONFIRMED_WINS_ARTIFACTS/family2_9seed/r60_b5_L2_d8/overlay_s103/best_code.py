def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256  # block size
    B = 5    # number of blocks
    L = 2    # window size
    W = world_size
    
    # Pre-compute which blocks this rank contributes to at each stage
    # Stage 0: initial sum (all blocks)
    # Stages 1-7: only blocks in this rank's window
    start_block = rank % B
    keep_blocks = set((start_block + j) % B for j in range(L))
    
    # Pre-compute coverage counts for each block at each stage
    # Stage 0: all W ranks contribute
    # Stages 1-7: only ranks whose window includes that block
    coverage = [[0] * B for _ in range(8)]
    
    # Stage 0: all ranks contribute all blocks
    for b in range(B):
        coverage[0][b] = W
    
    # Stages 1-7: count how many ranks cover each block
    for stage in range(1, 8):
        for r in range(W):
            r_start = r % B
            r_keep = set((r_start + j) % B for j in range(L))
            for b in r_keep:
                coverage[stage][b] += 1
    
    # Compute cumulative scaling factors
    # After stage s, block b has been scaled by product of 1/coverage[i][b] for i in 1..s
    cumulative_scale = [[1.0] * B for _ in range(8)]
    for b in range(B):
        cumulative_scale[0][b] = 1.0
        for stage in range(1, 8):
            cumulative_scale[stage][b] = cumulative_scale[stage-1][b] / coverage[stage][b]
    
    dtype = x.dtype
    
    # Strategy: Do all 8 all-reduces with pre-computed masks
    # Stage 0: all-reduce of x (no mask)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 1-7: mask and all-reduce, with cumulative scaling
    for stage in range(1, 8):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep_blocks:
                # Apply cumulative scaling from previous stages and prepare for this stage
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Apply scaling for this stage (except last stage doesn't need it since it's final)
        if stage < 7:
            for b in range(B):
                s[b*S:(b+1)*S] = s[b*S:(b+1)*S] / coverage[stage][b]
    
    return s