
def r68b_bidi_b022m5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b = [0.22 + 0.11*(r % 5) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        # Use torch.split for segment creation
        segs = torch.split(s, S)
        buf = s.clone() / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (segs[r] + b[r] * segs[r+1]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        segs = torch.split(s, S)
        for r in range(W - 2, -1, -1):
            segs[r] = segs[r] - b[r] * segs[r+1]
            s[r*S:(r+1)*S] = segs[r]
    
    segs = torch.split(s, S)
    buf = s.clone() / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (segs[r] + b[r] * segs[r+1]) / W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
