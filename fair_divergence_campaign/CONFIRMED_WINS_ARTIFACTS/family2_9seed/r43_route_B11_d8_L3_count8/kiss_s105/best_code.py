
def r43_route_B11_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 11; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Compute overlap counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Precompute inverse overlap counts
    inv_c_vals = [1.0 / c[b] if c[b] > 0 else 1.0 for b in range(B)]
    
    # Rank's window
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations with division
    for iteration in range(6):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                val = s[b*S:(b+1)*S]
                buf[b*S:(b+1)*S] = val
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Create inverse tensor and multiply in one shot
        inv_c = torch.zeros_like(acc)
        for b in range(B):
            inv_c[b*S:(b+1)*S] = inv_c_vals[b]
        s = acc * inv_c
    
    # Final iteration without division
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
