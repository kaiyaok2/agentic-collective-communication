
def r43_route_B11_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 11; L = 3; OFF = 2; STR = 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute this rank's window blocks
    start = (rank + OFF) % B
    
    # Zero out blocks not in window (in-place)
    for b in range(B):
        in_window = False
        for j in range(L):
            if b == (start + STR*j) % B:
                in_window = True
                break
        if not in_window:
            s[b*S:(b+1)*S] = 0.0
    
    # Final all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
