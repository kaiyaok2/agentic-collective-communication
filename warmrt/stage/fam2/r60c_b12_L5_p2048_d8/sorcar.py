
def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    OFF = 2
    
    # Initial all_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute which buckets this rank keeps
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(5))
    
    # Create buffer with selected buckets
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    # Final all_reduce (matching the last iteration which had no normalization)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return acc
