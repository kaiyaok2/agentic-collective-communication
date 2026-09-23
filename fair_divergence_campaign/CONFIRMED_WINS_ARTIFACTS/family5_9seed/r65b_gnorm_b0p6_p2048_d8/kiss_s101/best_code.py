
def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.6
    W = world_size
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Just final iteration (partial - no gr adjustment)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s
