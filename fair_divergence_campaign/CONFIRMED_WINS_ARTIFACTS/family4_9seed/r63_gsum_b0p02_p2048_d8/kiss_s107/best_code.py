
def r63_gsum_b0p02_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.02
    
    # Try with 2 all_reduce operations
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s + BETA * (s.sum())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    return acc
