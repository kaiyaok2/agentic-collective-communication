
def r65_gnorm_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    
    # Just the initial iteration
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, s / (1.0 + BETA * s.abs().mean())) / W
    
    return s
