
def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute accumulated weights for normalization
    A = [0.0] * 5
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = r % 5
        for j in range(3):
            A[(st + j) % 5] += w
    
    # Precompute which blocks this rank processes
    start = rank % 5
    keep = {(start + j) % 5 for j in range(3)}
    w = 0.5 + 0.02 * rank
    
    # Create mask once
    mask = torch.zeros(5 * S, device=x.device, dtype=x.dtype)
    for b in keep:
        mask[b*S:(b+1)*S] = w
    
    # Create normalization vector once
    norm = torch.zeros(5 * S, device=x.device, dtype=x.dtype)
    for b in range(5):
        norm[b*S:(b+1)*S] = 1.0 / A[b]
    
    # Apply 7 iterations
    for it in range(7):
        buf = s * mask
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        if it < 6:
            s = s * norm
    
    return s
