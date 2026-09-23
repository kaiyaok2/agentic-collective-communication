def r66_walsh_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    N = 16384
    
    # Precompute fixed sign vectors
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Initial all_reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pipelined dual-buffer approach:
    # We'll maintain two buffers and overlap computation with communication
    # by issuing the next all_reduce before completing all local arithmetic
    
    # Buffer 0: compute and dispatch first all_reduce
    buf0 = s + BETA * u * (v * s).mean()
    acc0 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Now we pipeline the remaining 6 iterations
    for i in range(6):
        # Complete current iteration arithmetic
        acc0 = acc0 / W
        s = acc0 - BETA * u * (v * acc0).mean()
        
        # Dispatch next all_reduce (overlapping with above if possible)
        buf1 = s + BETA * u * (v * s).mean()
        acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
        
        # Swap buffers for next iteration
        acc0 = acc1
    
    # Final iteration (no next buffer to dispatch)
    acc0 = acc0 / W
    s = acc0
    
    return s