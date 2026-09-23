
def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.6
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Normalize and scale before second reduce
    g = 1.0 + BETA * s.abs().mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s / (g * W))
    
    return acc
