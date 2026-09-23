
def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    
    # Compute count array once  
    c = [0]*10
    for r in range(W):
        st = (r + 2) % 10
        for j in range(4):
            c[(st + j) % 10] += 1
    
    # Compute keep set once
    start = (rank + 2) % 10
    keep = [(start + j) % 10 for j in range(4)]
    
    # Create scaling vector
    scale = torch.zeros(10 * S, device=x.device, dtype=x.dtype)
    for b in range(10):
        if c[b] > 0:
            scale[b*S:(b+1)*S] = 1.0 / c[b]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Try with 2 layers
    for _ in range(2):
        buf = torch.zeros_like(s)
        for b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        s = xm.all_reduce(xm.REDUCE_SUM, buf) * scale
    
    # Final layer without division
    buf = torch.zeros_like(s)
    for b in keep:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
