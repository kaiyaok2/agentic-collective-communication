
def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Create weight vectors - precompute scaled versions
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    a_scaled = [a / W for a in a_list]
    
    a_weights = torch.cat([torch.full((S,), a_scaled[r], device=x.device, dtype=x.dtype) 
                           for r in range(W)])
    
    # Compute inverse weights
    ones = torch.ones_like(a_weights)
    inv_weights = ones / (a_weights * W)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    s = s * a_weights
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    s = s * inv_weights
    
    # Iteration 2
    s = s * a_weights
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    s = s * inv_weights
    
    # Iteration 3
    s = s * a_weights
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
