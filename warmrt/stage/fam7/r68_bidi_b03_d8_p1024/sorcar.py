
def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b = [0.3 + 0.1*(r % 5) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Only forward pass, no backward
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
