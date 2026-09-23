
def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 16
    OFF = 2
    L = 5
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute which buckets this rank processes
    start = (rank + OFF) % B
    keep_list = [(start + j) % B for j in range(L)]
    
    # Create masked buffer by copying only kept buckets
    buf = torch.zeros_like(s)
    for b in keep_list:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    # Final all-reduce
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
