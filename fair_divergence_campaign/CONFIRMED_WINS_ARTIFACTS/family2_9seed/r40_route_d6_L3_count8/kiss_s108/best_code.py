
def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute overlap counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Build scale tensor using torch.cat
    scale_blocks = []
    for b in range(B):
        val = 1.0/c[b] if c[b] > 0 else 0.0
        block = torch.full((S,), val, device=x.device, dtype=x.dtype)
        scale_blocks.append(block)
    scale = torch.cat(scale_blocks, dim=0)
    
    # Compute window once
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # 4 iterations with normalization
    for _ in range(4):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * scale
    
    # Final iteration without normalization
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
