
def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; L = 3; OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create buffer by cloning and zeroing non-keep blocks
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    # Final all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
