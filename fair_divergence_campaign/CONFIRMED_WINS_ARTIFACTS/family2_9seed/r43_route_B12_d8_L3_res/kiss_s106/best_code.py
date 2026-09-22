
def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 12; W = world_size; L = 3; OFF = 2; STR = 1
    
    # All-reduce input
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute overlap counts and normalization factors
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Compute which blocks to keep
    start = (rank + OFF) % B
    keep = [(start + STR*j) % B for j in range(L)]
    
    # First mask
    buf = torch.zeros_like(s)
    for b in keep:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Normalize and mask simultaneously
    buf2 = torch.zeros_like(acc)
    for b in keep:
        buf2[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf2)
