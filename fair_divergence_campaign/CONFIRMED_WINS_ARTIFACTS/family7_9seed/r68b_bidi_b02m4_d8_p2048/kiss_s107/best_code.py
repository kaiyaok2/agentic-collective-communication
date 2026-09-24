
def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create coefficient tensor more efficiently
    b_expanded = torch.empty((W-1)*S, device=x.device, dtype=x.dtype)
    for r in range(W-1):
        b_expanded[r*S:(r+1)*S] = 0.2 + 0.12*(r % 4)
    
    # Vectorized computation
    buf = s.clone()
    buf[:-S] += b_expanded * s[S:]
    
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    return result / W
