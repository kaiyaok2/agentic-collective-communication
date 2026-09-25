def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    total_size = 5 * S
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute accumulation factors
    A = [0.0] * 5
    for r in range(world_size):
        w = 0.5 + 0.02 * r
        st = r % 5
        for j in range(3):
            A[(st + j) % 5] += w
    
    # This rank's parameters
    start = rank % 5
    keep_indices = [(start + j) % 5 for j in range(3)]
    w = 0.5 + 0.02 * rank
    
    # Create both masks in a single flat list
    combined_flat = []
    for b in range(5):
        val = w if b in keep_indices else 0.0
        val_norm = val / A[b] if val != 0 else 0.0
        combined_flat.extend([val_norm] * S)
    for b in range(5):
        val = w if b in keep_indices else 0.0
        combined_flat.extend([val] * S)
    
    combined = torch.tensor(combined_flat, device=x.device, dtype=x.dtype)
    mask_norm = combined[:total_size]
    mask = combined[total_size:]
    
    # First 6 iterations with normalized mask
    for _ in range(6):
        buf = s * mask_norm
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Last iteration with regular mask
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s