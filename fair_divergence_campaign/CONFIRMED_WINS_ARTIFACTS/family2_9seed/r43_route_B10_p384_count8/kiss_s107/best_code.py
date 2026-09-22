
def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 10
    L = 3
    OFF = 2
    STR = 1
    
    # This rank's window
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Zero out blocks not in window
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    # Final all_reduce
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
