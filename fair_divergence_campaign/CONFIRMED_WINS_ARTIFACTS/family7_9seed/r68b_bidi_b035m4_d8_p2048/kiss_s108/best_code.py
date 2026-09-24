
def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.35 + 0.1*(r % 4) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    inv_W = 1.0 / W
    
    # Just final forward and reduce (0 full iterations)
    buf = s * inv_W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
