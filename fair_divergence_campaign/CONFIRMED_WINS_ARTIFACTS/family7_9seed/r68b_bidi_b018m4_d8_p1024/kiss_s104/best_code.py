
def r68b_bidi_b018m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b = [0.18 + 0.13*(r % 4) for r in range(W)]
    
    # Initial reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 1 iteration with no backward sweep
    buf = s / W
    for r in range(W - 1):
        start = r * S
        end = (r + 1) * S
        next_end = (r + 2) * S
        buf[start:end] = (s[start:end] + b[r] * s[end:next_end]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
