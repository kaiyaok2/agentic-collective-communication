
def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 12; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Pre-compute overlap counts and scaling factors
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Pre-compute which blocks this rank keeps and which to zero
    start = (rank + OFF) % B
    keep_blocks = [(start + STR*j) % B for j in range(L)]
    zero_blocks = [b for b in range(B) if b not in keep_blocks]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Repeat 6 times: mask, all_reduce, scale
    for _ in range(6):
        buf = s.clone()
        for b in zero_blocks:
            buf[b*S:(b+1)*S] = 0.0
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            if c[b] > 0:
                s[b*S:(b+1)*S] = s[b*S:(b+1)*S] / c[b]
    
    # Final: mask, all_reduce, return  
    buf = s.clone()
    for b in zero_blocks:
        buf[b*S:(b+1)*S] = 0.0
    return xm.all_reduce(xm.REDUCE_SUM, buf)
