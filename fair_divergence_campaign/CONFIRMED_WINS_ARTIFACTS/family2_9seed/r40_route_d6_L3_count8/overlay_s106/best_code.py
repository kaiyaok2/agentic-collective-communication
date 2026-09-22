def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    dtype = x.dtype
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Determine this rank's window
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Create a mask tensor once
    mask = torch.zeros_like(x)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Create scale tensor once
    scale = torch.zeros_like(x)
    for b in range(B):
        scale[b*S:(b+1)*S] = 1.0 / c[b] if c[b] > 0 else 0.0
    
    # Dispatch 1: global all_reduce for sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Dispatches 2-6: Optimize by combining operations
    # Instead of 5 separate all_reduce calls, we can combine some logic
    for stage in range(5):
        # Apply mask (element-wise multiply is cheaper than loop)
        buf = s * mask
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by overlap count (except last stage)
        if stage < 4:
            s = acc * scale
        else:
            s = acc
    
    return s