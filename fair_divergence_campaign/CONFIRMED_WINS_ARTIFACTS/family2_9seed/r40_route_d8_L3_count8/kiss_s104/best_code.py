
def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute overlap counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Pre-compute this rank's window mask  
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Create plain mask
    mask = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Single iteration: use plain mask
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
