
def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    
    # Compute overlap counts once
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Create normalization vector once (avoid divide by zero)
    norm_vec = torch.ones(B * S, dtype=x.dtype, device=x.device)
    for b in range(B):
        if c[b] > 0:
            norm_vec[b*S:(b+1)*S] = c[b]
    
    # Compute this rank's window
    start = (rank + OFF) % B
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 rounds of windowed aggregation
    for round_idx in range(5):
        buf = torch.zeros_like(s)
        
        # Copy blocks in one operation if contiguous
        if start + L <= B:
            buf[start*S:(start+L)*S] = s[start*S:(start+L)*S]
        else:
            for j in range(L):
                b = (start + j) % B
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if round_idx < 4:
            acc = acc / norm_vec
        
        s = acc
    
    return s
