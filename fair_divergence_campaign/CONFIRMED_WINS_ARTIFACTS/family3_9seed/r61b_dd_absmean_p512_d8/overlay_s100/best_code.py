def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape once for all operations
    s_view = s.view(B, S)
    
    # 6 iterations of the pattern
    for iteration in range(6):
        # Compute absolute values once and reuse
        abs_s = s_view.abs()
        
        # Compute mean along dimension 1
        mean_abs = abs_s.mean(dim=1)
        
        # Compute factor
        f = 1.0 + mean_abs
        
        # Expand factor for broadcasting
        f_expanded = f.unsqueeze(1)
        
        # Scale
        buf_view = s_view * f_expanded
        
        # All-reduce
        buf = buf_view.reshape(-1)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Reshape and unscale
        acc_view = acc.view(B, S)
        s_view = acc_view / (world_size * f_expanded)
    
    # 7th iteration: similar pattern but without final unscaling
    abs_s = s_view.abs()
    mean_abs = abs_s.mean(dim=1)
    f = 1.0 + mean_abs
    f_expanded = f.unsqueeze(1)
    buf_view = s_view * f_expanded
    buf = buf_view.reshape(-1)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final division by world_size only
    acc = acc / world_size
    
    return acc