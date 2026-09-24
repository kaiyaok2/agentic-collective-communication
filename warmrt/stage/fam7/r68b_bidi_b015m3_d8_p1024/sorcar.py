
def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b_list = [0.15 + 0.1*(r % 3) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Forward sweep using torch.split
    buf = s / W
    s_parts = torch.split(s, S)
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s_parts[r] + b_list[r] * s_parts[r+1]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
