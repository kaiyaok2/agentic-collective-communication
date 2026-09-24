
def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.4 + 0.08*(r % 3) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Do 5 full iterations
    for iteration in range(5):
        # Forward sweep - combine divisions
        buf = s.clone()
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]
        buf = buf / W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward sweep
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
    
    # Final iteration
    buf = s.clone()
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]
    buf = buf / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
