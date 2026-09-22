
def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    B = 5
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute counts
    c = [0] * B
    for r in range(W):
        st = r % B
        c[st] += 1
        c[(st + 1) % B] += 1
    
    # Create division factors tensor
    div_factors = torch.zeros_like(s)
    for b in range(B):
        div_factors[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Determine which buckets this rank keeps and create mask
    start = rank % B
    keep0 = start
    keep1 = (start + 1) % B
    mask = torch.zeros_like(s)
    mask[keep0*S:(keep0+1)*S] = 1.0
    mask[keep1*S:(keep1+1)*S] = 1.0
    
    # Perform 6 full rounds with division
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * div_factors
    
    # Final round without division
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
