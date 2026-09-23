
def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute keep set for this rank
    start = (rank + 2) % 16
    keep = set((start + j) % 16 for j in range(5))
    
    # Clone and zero out unwanted buckets
    buf = s.clone()
    for b in range(16):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
