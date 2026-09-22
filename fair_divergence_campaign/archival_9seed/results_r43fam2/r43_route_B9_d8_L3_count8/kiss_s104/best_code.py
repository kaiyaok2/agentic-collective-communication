
def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 9
    L = 3
    OFF = 2
    STR = 1
    
    # This rank's blocks  
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Clone and zero out non-kept blocks
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0.0
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
