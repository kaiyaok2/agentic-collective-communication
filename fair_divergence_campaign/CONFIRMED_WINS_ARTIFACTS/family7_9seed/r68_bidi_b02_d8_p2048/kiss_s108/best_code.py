
def r68_bidi_b02_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.2 + 0.1*(r % 3) for r in range(W)]
    
    # Try 2 iterations
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
