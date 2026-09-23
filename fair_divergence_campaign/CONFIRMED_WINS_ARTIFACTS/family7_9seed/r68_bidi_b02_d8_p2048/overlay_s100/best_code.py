def r68_bidi_b02_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.2 + 0.1 * (r % 3) for r in range(W)]
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse stages 1-2
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    # Apply second forward coupling on buf
    buf2 = buf.clone()
    for r in range(W - 1):
        buf2[r*S:(r+1)*S] = buf[r*S:(r+1)*S] + b[r]*buf[(r+1)*S:(r+2)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf2)
    # Apply two backward couplings
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    # Fuse stages 3-4
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    buf2 = buf.clone()
    for r in range(W - 1):
        buf2[r*S:(r+1)*S] = buf[r*S:(r+1)*S] + b[r]*buf[(r+1)*S:(r+2)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf2)
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    # Fuse stages 5-6
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    buf2 = buf.clone()
    for r in range(W - 1):
        buf2[r*S:(r+1)*S] = buf[r*S:(r+1)*S] + b[r]*buf[(r+1)*S:(r+2)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf2)
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    # Stage 7 (single)
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s