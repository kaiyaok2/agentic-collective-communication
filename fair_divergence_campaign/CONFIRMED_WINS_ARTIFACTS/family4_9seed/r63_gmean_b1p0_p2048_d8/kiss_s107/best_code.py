
def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.0
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s + BETA * (s.mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s
