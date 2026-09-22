
def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 10; W = world_size; L = 3; OFF = 2; STR = 2
    
    # Pre-compute overlap counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Pre-compute which blocks this rank keeps
    start = (rank + OFF) % B
    keep_blocks = [(start + STR*j) % B for j in range(L)]
    
    # Create mask tensor once (1 for blocks we keep, 0 otherwise)
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in keep_blocks:
        mask[b*S:(b+1)*S] = 1.0
    
    # Create normalization factors tensor once (avoid division by zero)
    norm = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 7 iterations
    for iteration in range(7):
        buf = s * mask  # Mask using multiplication
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 6:
            s = acc * norm  # Normalize using multiplication
        else:
            s = acc
    
    return s
