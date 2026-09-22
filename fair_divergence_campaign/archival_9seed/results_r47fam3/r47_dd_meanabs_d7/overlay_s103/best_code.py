def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential all-reduce with local block-wise scaling.
    
    Performs 7 sequential all-reduce operations:
    1. Initial sum across all ranks
    2-7. Six iterative refinements where each computes scaled values and aggregates them
    
    Each all-reduce operates on the full 2048-element vector.
    Total: 7 all-reduce dispatches.
    """
    S = 256  # block size
    B = 8    # number of blocks
    dtype = x.dtype
    
    # Step 1: Initial all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Steps 2-7: Six iterative refinements
    for iteration in range(6):
        # Compute scaling factors for each block
        f = []
        for b in range(B):
            block = s[b*S:(b+1)*S]
            mean_abs = block.mean().abs()
            f.append(1.0 + mean_abs)
        
        # Scale each block by its factor
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        
        # All-reduce the scaled buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Unscale each block (divide by world_size * f[b])
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        
        # Update s for next iteration
        s = acc
    
    # Final step: compute scaling factors and apply them
    f = []
    for b in range(B):
        block = s[b*S:(b+1)*S]
        mean_abs = block.mean().abs()
        f.append(1.0 + mean_abs)
    
    # Scale each block by its factor
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # Final all-reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Divide by world_size only (no per-block unscaling)
    acc = acc / world_size
    
    return acc