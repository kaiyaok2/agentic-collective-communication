
def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; B = 8; W = world_size; L = 2; OFF = 2
    
    # Compute overlap counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # This rank's window
    start = (rank + OFF) % B
    b0 = (start) % B
    b1 = (start + 1) % B
    
    # Create mask
    mask = torch.zeros_like(s)
    mask[b0*S:(b0+1)*S] = 1.0
    mask[b1*S:(b1+1)*S] = 1.0
    
    # Single iteration without division
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
