
def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations with per-block division
    for _ in range(6):
        # Compute all block factors at once
        reshaped = s.view(B, S)
        f = 1.0 + reshaped.mean(dim=1).abs()
        
        # Apply factors using broadcasting
        f_expanded = f.unsqueeze(1).expand(B, S).reshape(-1)
        buf = s * f_expanded
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide using broadcasting
        s = acc / (world_size * f_expanded)
    
    # 7th iteration with simple division
    reshaped = s.view(B, S)
    f = 1.0 + reshaped.mean(dim=1).abs()
    
    f_expanded = f.unsqueeze(1).expand(B, S).reshape(-1)
    buf = s * f_expanded
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    return acc
