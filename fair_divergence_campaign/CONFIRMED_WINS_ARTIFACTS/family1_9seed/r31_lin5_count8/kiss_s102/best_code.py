def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Precompute weights as Python lists
    a = [1.0 + 0.25*(r % 5) for r in range(W)]
    
    # Create weight tensors once - vectorize the weights
    weight_fwd = []
    weight_inv = []
    for r in range(W):
        weight_fwd.extend([a[r] / W] * S)
        weight_inv.extend([1.0 / max(a[r], 1e-9)] * S)
    
    weight_fwd_t = torch.tensor(weight_fwd, device=x.device, dtype=x.dtype)
    weight_inv_t = torch.tensor(weight_inv, device=x.device, dtype=x.dtype)
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations of weighted aggregation with normalization
    for _ in range(6):
        buf = s * weight_fwd_t
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * weight_inv_t
    
    # Final iteration without normalization
    buf = s * weight_fwd_t
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s