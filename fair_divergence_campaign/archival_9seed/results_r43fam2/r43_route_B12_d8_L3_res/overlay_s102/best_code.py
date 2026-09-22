def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 12
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Phase 1: Initial all-reduce (dispatch 1)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Phase 2: Fuse the 7 iterations
    # We'll do 3 batched all-reduces instead of 7 separate ones
    # Group iterations: [0,1,2], [3,4], [5,6]
    
    # Iterations 0-2: batch together
    states = []
    for _ in range(3):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        states.append(buf)
    
    # Concatenate and do single all-reduce (dispatch 2)
    batched = torch.cat(states, dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, batched)
    
    # Split and process
    for i in range(3):
        chunk = reduced[i*S*B:(i+1)*S*B]
        for b in range(B):
            s[b*S:(b+1)*S] = chunk[b*S:(b+1)*S] / c[b]
    
    # Iterations 3-5: batch together
    states = []
    for _ in range(3):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        states.append(buf)
    
    # Concatenate and do single all-reduce (dispatch 3)
    batched = torch.cat(states, dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, batched)
    
    # Split and process
    for i in range(3):
        chunk = reduced[i*S*B:(i+1)*S*B]
        for b in range(B):
            s[b*S:(b+1)*S] = chunk[b*S:(b+1)*S] / c[b]
    
    # Iteration 6: final one
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s