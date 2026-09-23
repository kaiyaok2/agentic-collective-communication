
def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; B = 12; OFF = 2
    
    # Initial all_reduce to sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute which buckets this rank keeps
    start = (rank + OFF) % B
    keep_set = set((start + 1*j) % B for j in range(5))
    
    # Zero out non-kept buckets in-place
    for b in range(B):
        if b not in keep_set:
            s[b*S:(b+1)*S] = 0.0
    
    # Final all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
