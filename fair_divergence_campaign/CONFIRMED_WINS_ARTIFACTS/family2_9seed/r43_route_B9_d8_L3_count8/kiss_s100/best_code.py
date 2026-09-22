
def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 9; L = 3; OFF = 2; STR = 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Zero out blocks this rank doesn't need
    start = (rank + OFF) % B
    keep = {(start + j) % B for j in range(L)}
    
    result = s.clone()
    for b in range(B):
        if b not in keep:
            result[b*S:(b+1)*S] = 0.0
    
    # Final all_reduce
    return xm.all_reduce(xm.REDUCE_SUM, result)
