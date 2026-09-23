def r66b_walsh_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.7
    N = 16384
    dtype = x.dtype
    
    # Pre-compute fixed sign vectors once
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(dtype)
    
    # Pre-compute constant terms that can be reused
    u_beta = BETA * u
    
    # Initial all-reduce for sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 0: compute buf and launch all-reduce
    buf = s + u_beta * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Iterations 1-6: pipeline compute with communication
    for _ in range(6):
        # Update s from previous acc
        s = acc - u_beta * (v * acc).mean()
        
        # Compute next buffer (local computation)
        buf = s + u_beta * (v * s).mean()
        
        # Launch all-reduce while local compute is done
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
    
    # Final iteration: no need to compute next buffer
    s = acc
    
    return s