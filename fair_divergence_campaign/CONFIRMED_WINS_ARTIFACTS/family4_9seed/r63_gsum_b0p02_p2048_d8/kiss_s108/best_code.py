
def r63_gsum_b0p02_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.02
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    acc = xm.all_reduce(xm.REDUCE_SUM, s + BETA * s.sum())
    s = acc / W
    
    return s
