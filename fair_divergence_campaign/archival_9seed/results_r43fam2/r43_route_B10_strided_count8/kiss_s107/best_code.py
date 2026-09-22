
def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 10; W = world_size; L = 3; OFF = 2; STR = 2
    
    # Precompute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Precompute this rank's window
    start = (rank + OFF) % B
    keep = [(start + STR*j) % B for j in range(L)]
    
    # Create a reusable mask tensor
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Create normalization factors, avoiding division by zero
    norm = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations
    for it in range(7):
        # Apply mask using multiplication
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize on all but last iteration
        if it < 6:
            acc = acc * norm
        
        s = acc
    
    return s
