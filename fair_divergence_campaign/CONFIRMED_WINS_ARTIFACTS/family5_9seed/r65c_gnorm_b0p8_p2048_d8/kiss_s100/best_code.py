def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Just final iteration without the intermediate refinement
    g = 1.0 + BETA * s.abs().mean()
    s = xm.all_reduce(xm.REDUCE_SUM, s / g) / W
    
    return s