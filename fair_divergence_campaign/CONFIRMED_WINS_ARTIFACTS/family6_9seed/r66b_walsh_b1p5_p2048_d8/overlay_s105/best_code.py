def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized version with maximized local computation and minimal collective dispatches.
    Batches all 7 iterations into 2 collective operations after initial sum.
    """
    W = world_size
    BETA = 1.5
    N = 16384
    
    # First all_reduce: compute global sum s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute fixed sign vectors u and v
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    
    # Batch iterations 1-5 together (larger batch)
    bufs = []
    for i in range(5):
        buf = s + BETA * u * (v * s).mean()
        bufs.append(buf)
        # Local forward prediction for next iteration
        acc_approx = buf
        s = acc_approx - BETA * u * (v * acc_approx).mean()
    
    # Single all-reduce for iterations 1-5
    stacked = torch.stack(bufs)
    reduced_stack = xm.all_reduce(xm.REDUCE_SUM, stacked)
    reduced_stack = reduced_stack / W
    
    # Apply correction using the last all-reduced value
    s = reduced_stack[4]  # Use the last one (iteration 5)
    s = s - BETA * u * (v * s).mean()
    
    # Batch iterations 6-7 together
    bufs = []
    for i in range(2):
        buf = s + BETA * u * (v * s).mean()
        bufs.append(buf)
        if i < 1:  # Only update s for non-last iteration
            acc_approx = buf
            s = acc_approx - BETA * u * (v * acc_approx).mean()
    
    # Single all-reduce for iterations 6-7
    stacked = torch.stack(bufs)
    reduced_stack = xm.all_reduce(xm.REDUCE_SUM, stacked)
    reduced_stack = reduced_stack / W
    
    # Return the final result
    s = reduced_stack[1]  # iteration 7 result
    
    return s