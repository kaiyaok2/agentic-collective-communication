def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 13
    OFF = 2
    
    # Compute which buckets this rank keeps
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(4))
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Clone and zero out non-kept buckets
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s