def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 1.5
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 1-5: full normalization cycle
    for _ in range(5):
        s = s / (1.0 + BETA * s.abs().mean())
        s = xm.all_reduce(xm.REDUCE_SUM, s) / W
        A = s.abs().mean()
        s = s * (1.0 + BETA * A / (1.0 - BETA * A))
    
    # Final iteration: normalize and reduce only
    s = s / (1.0 + BETA * s.abs().mean())
    s = xm.all_reduce(xm.REDUCE_SUM, s) / W
    
    return s