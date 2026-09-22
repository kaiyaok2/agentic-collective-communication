
def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Mask directly without zeros_like
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Create masked version by zeroing non-keep blocks
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    return acc
