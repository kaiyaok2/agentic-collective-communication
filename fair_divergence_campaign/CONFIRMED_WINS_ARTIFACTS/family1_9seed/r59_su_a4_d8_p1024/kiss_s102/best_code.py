def r59_su_a4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    a = [1.0 + 0.6*(r % 4) for r in range(W)]
    
    # Only create forward weights
    weights_fwd = torch.cat([torch.full((S,), a[r] / W, device=x.device, dtype=x.dtype) 
                            for r in range(W)])
    
    # Pre-compute scaled weights for inverse operation
    weights_scaled = weights_fwd * W
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Repeat the iteration 6 times
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * weights_fwd)
        s = s / weights_scaled
    
    # Final weighted iteration
    s = xm.all_reduce(xm.REDUCE_SUM, s * weights_fwd)
    
    return s