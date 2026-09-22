
def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    L = 3
    OFF = 2
    
    # Compute overlap counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Create division factors tensor once  
    div_factors = torch.ones_like(x)
    for b in range(B):
        if c[b] > 0:
            div_factors[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Compute this rank's window mask once
    start = (rank + OFF) % B
    mask = torch.zeros_like(x)
    for j in range(L):
        b = (start + j) % B
        mask[b*S:(b+1)*S] = 1.0
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations: mask, all_reduce, divide
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * div_factors
    
    # Last iteration: mask, all_reduce only (no division)
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
