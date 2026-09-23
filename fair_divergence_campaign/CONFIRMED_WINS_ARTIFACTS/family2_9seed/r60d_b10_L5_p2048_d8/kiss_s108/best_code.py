
def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute keep set
    B = 10
    OFF = 2
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(5))
    
    # Single masked all_reduce
    buf = torch.zeros_like(s)
    for b in range(10):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
