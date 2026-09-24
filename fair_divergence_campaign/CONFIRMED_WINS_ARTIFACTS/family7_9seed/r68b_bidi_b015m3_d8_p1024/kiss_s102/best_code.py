
def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b = [0.15 + 0.1*(r % 3) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Single buffer approach
    buf = s / W
    
    # Create coefficient tensor
    b_tensor = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    for r in range(W - 1):
        b_tensor[r*S:(r+1)*S] = b[r]
    
    # Add weighted shifted contribution
    buf[:(W-1)*S] = buf[:(W-1)*S] + (b_tensor[:(W-1)*S] * s[S:]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
