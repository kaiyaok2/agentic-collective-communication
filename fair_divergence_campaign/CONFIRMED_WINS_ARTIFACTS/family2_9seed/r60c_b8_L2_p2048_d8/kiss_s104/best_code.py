
def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; B = 8; W = world_size; L = 2; OFF = 2
    
    # Pre-compute this rank's window
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Clone and zero out non-window blocks
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
