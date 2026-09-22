
def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 12
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute overlap counts per block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Determine which blocks this rank handles
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # Create mask tensor once (1.0 for blocks in window, 0.0 otherwise)
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Create overlap tensor once  
    overlap = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            overlap[b*S:(b+1)*S] = float(c[b])
    
    # 6 iterations with normalization using element-wise ops
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc / overlap
    
    # Final iteration without normalization
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
