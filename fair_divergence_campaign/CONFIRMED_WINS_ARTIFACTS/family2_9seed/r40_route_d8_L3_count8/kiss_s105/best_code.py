
def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; L = 3; OFF = 2
    
    # Initial all_reduce to get sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Clone and zero out blocks not in this rank's window
    buf = s.clone()
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
