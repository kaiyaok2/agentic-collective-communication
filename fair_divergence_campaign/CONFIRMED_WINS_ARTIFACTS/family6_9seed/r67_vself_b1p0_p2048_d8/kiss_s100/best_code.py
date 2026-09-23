
def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.0
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute alternating sum directly
    s_even = s[0::2]
    s_odd = s[1::2]
    alt_mean = (s_even.sum() - s_odd.sum()) / N
    
    # Apply correction: add BETA * alt_mean to even indices, subtract from odd
    buf = s.clone()
    buf[0::2] = s_even + BETA * alt_mean
    buf[1::2] = s_odd - BETA * alt_mean
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    return acc
