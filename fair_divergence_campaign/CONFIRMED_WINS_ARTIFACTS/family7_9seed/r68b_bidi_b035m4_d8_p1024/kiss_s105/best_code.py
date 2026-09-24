
def r68b_bidi_b035m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b = [0.35 + 0.09*(r % 4) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s.clone()
    
    chunks = torch.split(s, S)
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = chunks[r] + b[r] * chunks[r+1]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s / W
