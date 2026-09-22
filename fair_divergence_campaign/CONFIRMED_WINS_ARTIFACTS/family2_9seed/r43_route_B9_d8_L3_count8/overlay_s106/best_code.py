def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 9
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Compute per-block overlap count
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Pack all 7 rounds into fewer dispatches
    # Strategy: do all 7 rounds in a single mega-batch
    states = [s]
    for round_idx in range(6):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = states[-1][b*S:(b+1)*S]
        states.append(buf)
    
    # Concatenate all 7 intermediate states
    packed = torch.cat(states[1:], dim=0)
    
    # Single all-reduce for all 7 rounds
    reduced = xm.all_reduce(xm.REDUCE_SUM, packed)
    
    # Unpack and apply scaling sequentially
    for round_idx in range(6):
        offset = round_idx * B * S
        acc = reduced[offset:offset + B*S]
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    # Final round (7th)
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s