def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute kept buckets
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(4))
    
    # Just do the final selection and all_reduce
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    return xm.all_reduce(xm.REDUCE_SUM, buf)