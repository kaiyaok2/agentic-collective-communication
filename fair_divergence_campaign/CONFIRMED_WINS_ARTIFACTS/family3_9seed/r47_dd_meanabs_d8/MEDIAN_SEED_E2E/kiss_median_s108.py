
def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 1-6
    for iteration in range(6):
        # Compute factors vectorized
        s_reshaped = s.view(B, S)
        factors = 1.0 + s_reshaped.mean(dim=1).abs()
        
        # Expand factors using repeat
        factor_tensor = factors.repeat_interleave(S)
        
        # Scale, reduce, unscale
        buf = s * factor_tensor
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc / (world_size * factor_tensor)
    
    # Iteration 7
    s_reshaped = s.view(B, S)
    factors = 1.0 + s_reshaped.mean(dim=1).abs()
    factor_tensor = factors.repeat_interleave(S)
    
    buf = s * factor_tensor
    return xm.all_reduce(xm.REDUCE_SUM, buf) / world_size
