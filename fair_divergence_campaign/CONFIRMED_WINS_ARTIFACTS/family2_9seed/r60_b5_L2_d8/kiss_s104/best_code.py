
def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts - how many ranks keep each bucket
    c = [0] * 5
    for r in range(W):
        st = r % 5
        ks = set((st + 1*j) % 5 for j in range(2))
        for b in ks:
            c[b] += 1
    
    # Precompute keep set for this rank (stays constant across iterations)
    start = rank % 5
    keep = set((start + 1*j) % 5 for j in range(2))
    
    # Create weight vector for efficient division (replaces 5 slices with 1 multiply)
    weight_list = [1.0/c[b] for b in range(5)]
    weights = torch.tensor([w for w in weight_list for _ in range(S)], 
                          device=x.device, dtype=x.dtype)
    
    # 7 iterations
    for iteration in range(7):
        buf = torch.zeros_like(s)
        for b in range(5):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by counts (except last iteration)
        if iteration < 6:
            s = acc * weights
        else:
            s = acc
    
    return s
