def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    
    dtype = x.dtype
    
    # Step 1: Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Step 2: Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 3: Reduce-scatter approach
    # Each rank will be responsible for computing final values for certain blocks
    # We'll distribute the 8 blocks across ranks
    
    # Determine which blocks this rank will process
    blocks_per_rank = B // W if B >= W else 1
    extra_blocks = B % W if B >= W else 0
    
    if B >= W:
        # Standard case: distribute blocks across ranks
        if rank < extra_blocks:
            my_blocks = list(range(rank * (blocks_per_rank + 1), (rank + 1) * (blocks_per_rank + 1)))
        else:
            start_block = extra_blocks * (blocks_per_rank + 1) + (rank - extra_blocks) * blocks_per_rank
            my_blocks = list(range(start_block, start_block + blocks_per_rank))
    else:
        # More ranks than blocks: only some ranks process blocks
        my_blocks = [rank] if rank < B else []
    
    # Step 4: Perform reduce-scatter-like operation
    # Each rank collects contributions for its assigned blocks through the 5 iterations
    result = s.clone()
    
    for iteration in range(5):
        # Determine which blocks this rank contributes to in this iteration
        start = (rank + OFF) % B
        keep = set((start + j) % B for j in range(L))
        
        # Create buffer with only the blocks this rank should contribute
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = result[b*S:(b+1)*S]
        
        # All-reduce to sum contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by overlap counts (skip on last iteration as per reference)
        if iteration < 4:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        result = acc
    
    return result