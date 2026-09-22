
def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 12; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Pre-compute this rank's window
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # Initial all-reduce to establish global state
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create mask by zeroing out blocks not in window
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    # Final all-reduce of masked data
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
