
def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 3.0
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for i in range(6):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        
        if i < 5:
            A = acc.abs().mean()
            M = A / (1.0 - BETA * A)
            gr = 1.0 + BETA * M
            s = acc * gr
        else:
            s = acc
    
    return s
