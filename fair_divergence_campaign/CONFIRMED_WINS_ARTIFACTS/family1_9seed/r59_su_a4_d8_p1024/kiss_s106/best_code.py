
def r59_su_a4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Pre-compute weights as Python list
    a = [1.0 + 0.6*(r % 4) for r in range(W)]
    
    # Create weight tensors once - vectorized approach
    fwd_weights = torch.tensor([a[r] / W for r in range(W) for _ in range(S)], 
                                device=x.device, dtype=x.dtype)
    inv_weights = torch.tensor([1.0 / max(a[r], 1e-9) for r in range(W) for _ in range(S)], 
                                device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations with vectorized operations
    for i in range(7):
        s = s * fwd_weights
        s = xm.all_reduce(xm.REDUCE_SUM, s)
        if i < 6:  # Don't apply inverse weights on last iteration
            s = s * inv_weights
    
    return s
