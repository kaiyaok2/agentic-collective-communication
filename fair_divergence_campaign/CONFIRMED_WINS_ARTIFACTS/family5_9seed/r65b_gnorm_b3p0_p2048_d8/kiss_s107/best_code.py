def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 3.0
    
    # First all_reduce for synchronization
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 1-6 are mathematically no-ops: s = (s/g)*g = s
    # Only iteration 7 matters
    X = s.abs().mean()
    g = 1.0 + BETA * X
    s = s / g
    
    return s