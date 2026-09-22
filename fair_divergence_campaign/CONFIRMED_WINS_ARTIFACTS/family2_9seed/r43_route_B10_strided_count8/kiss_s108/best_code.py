
def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 10; L = 3; OFF = 2; STR = 2
    
    # Precompute this rank's window once
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Try using clone and zeroing out non-keep blocks
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    return result
