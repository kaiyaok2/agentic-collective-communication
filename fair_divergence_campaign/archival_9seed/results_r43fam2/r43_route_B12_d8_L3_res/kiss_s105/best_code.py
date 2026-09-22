
def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 12; W = world_size; L = 3; OFF = 2; STR = 1
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute overlap counts once
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Create division vector once
    div_vec = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            div_vec[b*S:(b+1)*S] = c[b]
    
    # Compute this rank's window once
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # 7 iterations - first 6 with division
    for iteration in range(6):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc / div_vec  # Vectorized division
    
    # Final iteration without division
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
