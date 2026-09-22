
def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 9; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute which blocks this rank keeps
    start = (rank + OFF) % B
    keep_blocks = set((start + STR*j) % B for j in range(L))
    
    # Precompute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Build mask tensor once (1.0 for kept blocks, 0.0 otherwise)
    mask = torch.zeros_like(s)
    for b in keep_blocks:
        mask[b*S:(b+1)*S] = 1.0
    
    # Build overlap tensor once (overlap count for each element, avoid div by zero)
    overlap = torch.zeros_like(s)
    for b in range(B):
        overlap[b*S:(b+1)*S] = float(max(c[b], 1))
    
    # 6 iterations with division
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc / overlap
    
    # Last iteration without division
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
