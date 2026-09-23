
def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    result = s / (1.0 + BETA * s.abs().mean())
    
    return result
