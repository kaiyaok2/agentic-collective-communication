
def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; B = 8; W = world_size; L = 2; OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Clone and zero out blocks not in window
    start = (rank + OFF) % B
    buf = s.clone()
    
    for b in range(B):
        if b != start and b != (start + 1) % B:
            buf[b*S:(b+1)*S] = 0.0
    
    # Final all_reduce
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
