
def r68b_bidi_b018m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    W_inv = 1.0 / W
    
    # Create shifted version - pad at end
    s_shifted = torch.cat([s[S:], s[:S] * 0])
    
    # Create coefficient tensor
    b_tensor = torch.zeros_like(s)
    for r in range(W - 1):
        b_r = 0.18 + 0.13 * (r % 4)
        b_tensor[r*S:(r+1)*S] = b_r
    
    # Apply transformation
    buf = (s + b_tensor * s_shifted) * W_inv
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
