
def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute accumulated weights
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = r % 6
        for j in range(5):
            A[(st + j) % 6] += w
    
    # This rank's parameters
    start = rank % 6
    w = 0.5 + 0.02 * rank
    skip_block = (start + 5) % 6
    
    # Create weighted mask with normalization built in for first 6 iterations
    weighted_mask_norm = torch.zeros(6 * S, device=x.device, dtype=x.dtype)
    for b in range(6):
        if b != skip_block:
            weighted_mask_norm[b*S:(b+1)*S] = w / A[b]
    
    # Create weighted mask for last iteration (no normalization)
    weighted_mask_final = torch.full((6 * S,), w, device=x.device, dtype=x.dtype)
    weighted_mask_final[skip_block * S:(skip_block + 1) * S] = 0.0
    
    # First 6 iterations with built-in normalization
    for iteration in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * weighted_mask_norm)
    
    # Last iteration without normalization
    s = xm.all_reduce(xm.REDUCE_SUM, s * weighted_mask_final)
    
    return s
