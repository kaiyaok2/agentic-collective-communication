
def r68b_bidi_b035m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b_vals = [0.35 + 0.09*(r % 4) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create coefficient tensor for vectorized operation
    b = torch.zeros(W * S, dtype=x.dtype, device=x.device)
    for r in range(W - 1):
        b[r*S:(r+1)*S] = b_vals[r]
    
    # Vectorized computation
    s_shifted = torch.cat([s[S:], torch.zeros(S, dtype=x.dtype, device=x.device)])
    buf = (s + b * s_shifted) / W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
