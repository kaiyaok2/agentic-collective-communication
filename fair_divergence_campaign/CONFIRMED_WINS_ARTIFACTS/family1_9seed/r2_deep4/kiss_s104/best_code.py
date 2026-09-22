def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Create base weights tensor
    weights = torch.tensor([1.0 + 0.5*(r % 3) for r in range(W) for _ in range(S)], 
                           device=x.device, dtype=x.dtype)
    scaled_weights = weights / W
    inv_weights = weights ** (-1.0)
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    s = xm.all_reduce(xm.REDUCE_SUM, scaled_weights * s) * inv_weights
    
    # Iteration 2
    s = xm.all_reduce(xm.REDUCE_SUM, scaled_weights * s) * inv_weights
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, scaled_weights * s)
    
    return s