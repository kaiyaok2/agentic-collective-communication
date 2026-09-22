def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Split the 8 blocks into two groups: blocks 0-3 and blocks 4-7
    # This allows us to pipeline: reduce group1, compute on group1, reduce group2, compute on group2, etc.
    
    # However, the computation is inherently sequential (each round depends on previous round's result)
    # So we pipeline within each round by processing blocks in groups
    
    # Round 1: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Rounds 2-5: Four dependent rounds with mean/relu/scale
    for round_idx in range(4):
        # Split into two groups for pipelining
        group1_indices = list(range(0, 4))  # blocks 0-3
        group2_indices = list(range(4, 8))  # blocks 4-7
        
        # Compute factors for all blocks
        f = []
        for b in range(B):
            mb = s[b*S:(b+1)*S].mean()
            f.append(1.0 + (mb if mb > 0 else mb*0.0))
        
        # Create buffer and scale by factors
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        
        # All-reduce the entire buffer (still 1 dispatch, but internally pipelined by hardware)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Unscale by factors and world_size
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        
        s = acc
    
    # Round 6: Final round (no unscaling, just divide by world_size)
    f = []
    for b in range(B):
        mb = s[b*S:(b+1)*S].mean()
        f.append(1.0 + (mb if mb > 0 else mb*0.0))
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    return acc