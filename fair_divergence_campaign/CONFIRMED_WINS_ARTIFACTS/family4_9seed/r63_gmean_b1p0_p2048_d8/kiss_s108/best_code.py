
def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.0
    
    # Initial all_reduce - after this, s is identical on all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Since s is now globally consistent, remaining operations are local
    for _ in range(6):
        buf = s + BETA * s.mean()
        # all_reduce of identical values: SUM gives W * buf, then / W = buf
        s = buf - 0.5 * buf.mean()
    
    # Final iteration
    buf = s + BETA * s.mean()
    s = buf
    
    return s
