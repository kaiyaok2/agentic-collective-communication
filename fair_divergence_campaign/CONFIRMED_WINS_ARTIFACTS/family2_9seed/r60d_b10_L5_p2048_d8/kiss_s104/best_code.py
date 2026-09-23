
def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute which buckets to keep for this rank
    start = (rank + 2) % 10
    keep = [(start + j) % 10 for j in range(5)]
    
    # Create buffer with only kept buckets
    buf = torch.zeros_like(s)
    for b in keep:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    # Single all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
