
def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 10; W = world_size; L = 3; OFF = 2; STR = 2
    
    # Compute overlap counts once
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Create mask tensor once - identifies this rank's window
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    mask = torch.zeros(B*S, dtype=x.dtype, device=x.device)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Create normalization tensor once - per-block division factors
    # Use 1.0 for blocks with no coverage to avoid division by zero
    norm = torch.ones(B*S, dtype=x.dtype, device=x.device)
    for b in range(B):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations - use multiplication instead of slicing
    for iter in range(7):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iter < 6:
            s = acc * norm
        else:
            s = acc
    
    return s
