
def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    
    # Determine which buckets this rank keeps
    start = (rank + 2) % B
    keep = {(start + j) % B for j in range(5)}
    
    # All-reduce input
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Mask the data
    for b in range(B):
        if b not in keep:
            s[b*S:(b+1)*S] = 0
    
    # Final all_reduce
    return xm.all_reduce(xm.REDUCE_SUM, s)
