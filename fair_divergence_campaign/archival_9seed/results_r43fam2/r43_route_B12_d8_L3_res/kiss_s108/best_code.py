def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 12; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Compute overlap counts and create normalization vector
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Create normalization vector once
    norm_data = []
    for b in range(B):
        if c[b] > 0:
            norm_data.extend([1.0 / c[b]] * S)
        else:
            norm_data.extend([1.0] * S)
    norm_vec = torch.tensor(norm_data, device=x.device, dtype=x.dtype)
    
    # This rank's window
    start = (rank + OFF) % B
    keep = [(start + STR*j) % B for j in range(L)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations
    for iter_idx in range(7):
        # Create buffer and mask
        buf = torch.zeros_like(s)
        for b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize using vector multiplication (skip on last iteration)
        if iter_idx < 6:
            acc = acc * norm_vec
        
        s = acc
    
    return s