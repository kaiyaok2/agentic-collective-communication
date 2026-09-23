def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.6
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Single pass - normalize and all_reduce
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s