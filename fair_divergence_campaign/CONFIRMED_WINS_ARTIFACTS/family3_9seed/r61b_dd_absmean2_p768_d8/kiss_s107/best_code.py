
def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768; B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with per-batch normalization
    for _ in range(6):
        # Compute factors vectorized
        s_reshaped = s.view(B, S)
        factors = 1.0 + 2.0 * s_reshaped.abs().mean(dim=1)
        factors_expanded = factors.unsqueeze(1).expand(B, S).reshape(-1)
        
        # Vectorized multiplication
        buf = s * factors_expanded
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Vectorized division
        acc = acc / (world_size * factors_expanded)
        s = acc
    
    # Last iteration
    s_reshaped = s.view(B, S)
    factors = 1.0 + 2.0 * s_reshaped.abs().mean(dim=1)
    factors_expanded = factors.unsqueeze(1).expand(B, S).reshape(-1)
    
    buf = s * factors_expanded
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    s = acc
    
    return s
