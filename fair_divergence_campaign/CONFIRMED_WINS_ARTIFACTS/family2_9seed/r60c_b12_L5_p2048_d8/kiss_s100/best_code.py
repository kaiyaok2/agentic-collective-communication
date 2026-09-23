
def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    OFF = 2
    
    # Compute counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(5):
            c[(st + j) % B] += 1
    
    # Compute keep set
    start = (rank + OFF) % B
    keep_set = set((start + j) % B for j in range(5))
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First iteration - zero out non-kept buckets
    for b in range(B):
        if b not in keep_set:
            s[b*S:(b+1)*S] = 0
    
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    # Divide by counts
    for b in range(B):
        if c[b] > 0:
            s[b*S:(b+1)*S] /= c[b]
    
    # Final iteration - zero out non-kept buckets again
    for b in range(B):
        if b not in keep_set:
            s[b*S:(b+1)*S] = 0
    
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
