
def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 13
    OFF = 2
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute keep buckets
    start = (rank + OFF) % B
    keep = [(start + j) % B for j in range(4)]
    
    # Extract only keep buckets, zero others in-place
    result = torch.zeros_like(s)
    for b in keep:
        result[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, result)
    
    return s
