def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Baseline Sequential All-Reduce Chain:
    Execute 6 dependent all_reduce operations sequentially as in the seed implementation.
    
    Pattern:
    1. First all_reduce: sum x across ranks
    2. For iterations 2-5: compute scaling factors f[b] = 1 + ReLU(mean(block b)),
       scale each block, all_reduce sum, then unscale by dividing
    3. Iteration 6: same as 2-5 but final division is by world_size only (no f factor)
    """
    S = 256
    B = 8
    dtype = x.dtype
    
    # Iteration 1: Initial all_reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 2-5: scaling with f factors
    for iteration in range(4):
        # Compute scaling factors f[b] = 1 + ReLU(mean(block b))
        f = []
        for b in range(B):
            block = s[b*S:(b+1)*S]
            mb = block.mean()
            f_b = 1.0 + (mb if mb > 0 else mb * 0.0)
            f.append(f_b)
        
        # Scale each block by f[b]
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        
        # All-reduce sum
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Unscale: divide by (world_size * f[b])
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        
        s = acc
    
    # Iteration 6: final iteration (scaling but simpler final division)
    f = []
    for b in range(B):
        block = s[b*S:(b+1)*S]
        mb = block.mean()
        f_b = 1.0 + (mb if mb > 0 else mb * 0.0)
        f.append(f_b)
    
    # Scale each block by f[b]
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # All-reduce sum
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final division: by world_size only
    acc = acc / world_size
    
    s = acc
    
    return s