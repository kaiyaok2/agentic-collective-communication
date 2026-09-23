
def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 3.0
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Just the final iteration
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s
