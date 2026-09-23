def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    
    # Compute keep buckets
    start = (rank + 2) % 12
    keep = sorted([(start + j) % 12 for j in range(5)])
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Check if keep buckets are contiguous
    if keep[-1] - keep[0] == 4:  # Contiguous
        # Copy in one go
        buf = torch.zeros_like(s)
        buf[keep[0]*S:(keep[-1]+1)*S] = s[keep[0]*S:(keep[-1]+1)*S]
    else:
        # Non-contiguous, copy individually
        buf = torch.zeros_like(s)
        for b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    # Final all reduce
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result