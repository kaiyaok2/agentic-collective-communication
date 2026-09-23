
def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 13
    OFF = 2
    W = world_size
    
    # Precompute counts once
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(4):
            c[(st + j) % B] += 1
    
    # Create count tensor for vectorized division
    c_list = []
    for b in range(B):
        c_list.extend([c[b]] * S)
    c_tensor = torch.tensor(c_list, device=x.device, dtype=x.dtype)
    
    # Precompute which buckets to keep
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(4))
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 full iterations with division
    for iteration in range(6):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc / c_tensor
    
    # Last iteration without division
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
