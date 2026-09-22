
def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts once
    c = [0] * 5
    for r in range(W):
        st = r % 5
        c[st] += 1
        c[(st + 1) % 5] += 1
    
    # Create division tensor once
    div_tensor = torch.zeros(1280, device=x.device, dtype=x.dtype)
    for b in range(5):
        div_tensor[b*S:(b+1)*S] = c[b]
    
    # Compute keep set once for this rank
    start = rank % 5
    keep = {start, (start + 1) % 5}
    
    # Perform 7 iterations
    for iteration in range(7):
        buf = torch.zeros_like(s)
        for b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 6:
            s = s / div_tensor
    
    return s
