def r59b_su_a4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Precompute weights as Python list
    a = [1.0 + 0.6*(r % 4) for r in range(W)]
    
    # Build weight tensors by stacking repeated values
    weight_chunks = []
    inv_weight_chunks = []
    for r in range(W):
        w_val = a[r] / W
        inv_w_val = 1.0 / max(a[r], 1e-9)
        weight_chunks.append(torch.full((S,), w_val, device=x.device, dtype=x.dtype))
        inv_weight_chunks.append(torch.full((S,), inv_w_val, device=x.device, dtype=x.dtype))
    
    weights = torch.cat(weight_chunks)
    inv_weights = torch.cat(inv_weight_chunks)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations with normalization
    for iteration in range(6):
        buf = weights * s
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * inv_weights
    
    # Final iteration without normalization
    buf = weights * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
