def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Iteration 0: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-reshape once
    s_reshaped = s.view(B, S)
    
    # Pre-compute constants
    inv_S = 1.0 / S
    inv_world_size = 1.0 / world_size
    
    # Iterations 1-6: fused operations with reduced dispatches
    for i in range(6):
        # Fused computation: mean(s^2) in single expression
        f = 1.0 + (s_reshaped * s_reshaped).sum(dim=1) * inv_S  # Shape: [B]
        
        # Fused scale and flatten (avoiding intermediate buf variable)
        buf_flat = (s_reshaped * f.view(B, 1)).view(-1)
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf_flat)
        
        # Fused divide and reshape
        s_reshaped = (acc.view(B, S) * inv_world_size) / f.view(B, 1)
    
    # Iteration 7 (final): fused final computation
    f = 1.0 + (s_reshaped * s_reshaped).sum(dim=1) * inv_S
    buf_flat = (s_reshaped * f.view(B, 1)).view(-1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf_flat)
    s = acc * inv_world_size
    
    return s