
def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384; B = 10; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Compute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Create mask for this rank's window
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    start = (rank + OFF) % B
    for j in range(L):
        b = (start + STR*j) % B
        mask[b*S:(b+1)*S] = 1.0
    
    # Create division factors tensor (avoid division by zero)
    div_factors = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            div_factors[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with division
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * div_factors
    
    # Final iteration without division
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
