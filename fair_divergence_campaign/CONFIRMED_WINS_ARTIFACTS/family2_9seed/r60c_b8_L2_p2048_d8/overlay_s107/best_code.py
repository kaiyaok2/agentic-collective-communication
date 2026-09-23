def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 8
    W = world_size
    L = 2
    OFF = 2
    
    # Phase 1: Get global sum with a single all-reduce
    global_sum = xm.all_reduce(xm.REDUCE_SUM, x.clone())
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Phase 2: Iteratively compute 8 intermediate states locally
    # Start with state 0 = global_sum
    states = [global_sum.clone()]
    
    # For each of the 7 subsequent states (iterations 1-7)
    for iteration in range(1, 8):
        # Start from previous state
        prev_state = states[-1]
        
        # Apply mask+scale: each rank zeros out its window and scales rest by overlap counts
        start = (rank + OFF) % B
        keep = set((start + j) % B for j in range(L))
        
        buf = torch.zeros_like(prev_state)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = prev_state[b*S:(b+1)*S]
        
        # All-reduce to aggregate masked contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by overlap counts (except for last iteration which doesn't need scaling)
        if iteration < 7:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        states.append(acc)
    
    # Return the 8th state (last one)
    return states[7]