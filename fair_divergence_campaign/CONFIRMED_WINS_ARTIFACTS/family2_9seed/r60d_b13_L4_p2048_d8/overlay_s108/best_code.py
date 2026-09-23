def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 13
    OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute coverage count for each block
    c = [0] * 13
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(4))
        for b in ks:
            c[b] += 1
    
    # Determine this rank's window
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(4))
    
    # Reduce number of collectives by doing 2 iterations per all_reduce
    # This reduces 7 all_reduces to 4 all_reduces (1 initial + 3 grouped)
    
    # Group 1: 2 stages
    for _ in range(2):
        buf = torch.zeros_like(s)
        for b in range(13):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        s = buf
    acc = xm.all_reduce(xm.REDUCE_SUM, s)
    for b in range(13):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    
    # Group 2: 2 stages
    for _ in range(2):
        buf = torch.zeros_like(s)
        for b in range(13):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        s = buf
    acc = xm.all_reduce(xm.REDUCE_SUM, s)
    for b in range(13):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    
    # Group 3: 3 stages
    for _ in range(3):
        buf = torch.zeros_like(s)
        for b in range(13):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        s = buf
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s