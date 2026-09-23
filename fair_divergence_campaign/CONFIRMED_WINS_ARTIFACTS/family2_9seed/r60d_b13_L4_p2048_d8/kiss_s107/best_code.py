def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 13
    OFF = 2
    
    # Pre-compute keep set
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(4))
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Mask in-place by zeroing out non-kept buckets
    for b in range(B):
        if b not in keep:
            s[b*S:(b+1)*S] = 0
    
    # Final all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s