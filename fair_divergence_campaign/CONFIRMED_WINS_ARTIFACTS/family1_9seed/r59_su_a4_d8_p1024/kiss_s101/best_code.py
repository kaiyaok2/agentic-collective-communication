def r59_su_a4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Pre-compute weight vectors (vectorized approach)
    a_weights = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    inv_a_weights = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    
    for r in range(W):
        a_val = 1.0 + 0.6 * (r % 4)
        a_weights[r*S:(r+1)*S] = a_val / W
        inv_a_weights[r*S:(r+1)*S] = 1.0 / max(a_val, 1e-9)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterative weighted averaging (6 full iterations)
    for _ in range(6):
        buf = a_weights * s
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * inv_a_weights
    
    # Final weighted sum (no inverse scaling)
    buf = a_weights * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s