
def r68b_bidi_b035m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create shifted version (neighbor values)
    s_shifted = torch.zeros_like(s)
    s_shifted[:(W-1)*S] = s[S:W*S]
    
    # Create weight vector
    b_vec = torch.zeros_like(s)
    for r in range(W - 1):
        b_vec[r*S:(r+1)*S] = 0.35 + 0.09 * (r % 4)
    
    # Compute weighted combination
    buf = (s + b_vec * s_shifted) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
