
def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 5
    
    # First all_reduce to sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Mask in place by zeroing out non-kept buckets
    start = rank % B
    keep = set((start + j) % B for j in range(2))
    
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    # Second all_reduce
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
