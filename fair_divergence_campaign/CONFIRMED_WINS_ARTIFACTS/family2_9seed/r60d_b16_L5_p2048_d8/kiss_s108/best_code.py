
def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 16
    OFF = 2
    L = 5
    
    # Initial all_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine which buckets this rank keeps
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Create buffer and copy kept buckets
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0.0
    
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
