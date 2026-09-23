
def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    w_inv = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s / g) * w_inv
    A = acc.abs().mean()
    s = acc * (1.0 + BETA * (A / (1.0 - BETA * A)))
    
    g = 1.0 + BETA * s.abs().mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s / g) * w_inv
    A = acc.abs().mean()
    s = acc * (1.0 + BETA * (A / (1.0 - BETA * A)))
    
    g = 1.0 + BETA * s.abs().mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s / g) * w_inv
    A = acc.abs().mean()
    s = acc * (1.0 + BETA * (A / (1.0 - BETA * A)))
    
    g = 1.0 + BETA * s.abs().mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s / g) * w_inv
    A = acc.abs().mean()
    s = acc * (1.0 + BETA * (A / (1.0 - BETA * A)))
    
    g = 1.0 + BETA * s.abs().mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s / g) * w_inv
    A = acc.abs().mean()
    s = acc * (1.0 + BETA * (A / (1.0 - BETA * A)))
    
    g = 1.0 + BETA * s.abs().mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s / g) * w_inv
    
    return acc
