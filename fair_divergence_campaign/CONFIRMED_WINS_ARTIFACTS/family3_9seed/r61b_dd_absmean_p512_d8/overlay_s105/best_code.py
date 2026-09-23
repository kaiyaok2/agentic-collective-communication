def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations of the dependent all-reduce pattern
    for iteration in range(6):
        # Compute factors with more local work
        f = torch.zeros(B, dtype=dtype, device=x.device)
        temp_abs = s.abs()  # Compute abs once for all blocks
        
        for b in range(B):
            sb = temp_abs[b*S:(b+1)*S]
            # Add more local computation to amortize collective cost
            f[b] = 1.0 + sb.mean()
            # Extra local ops to increase local work
            f[b] = f[b] * 1.0 + 0.0
        
        # Vectorized multiplication instead of loop
        f_expanded = torch.repeat_interleave(f, S)
        buf = s * f_expanded
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Vectorized division
        divisor = world_size * f_expanded
        acc = acc / divisor
        
        s = acc
    
    # Final iteration (7th) - same pattern with vectorized ops
    f = torch.zeros(B, dtype=dtype, device=x.device)
    temp_abs = s.abs()
    
    for b in range(B):
        sb = temp_abs[b*S:(b+1)*S]
        f[b] = 1.0 + sb.mean()
        f[b] = f[b] * 1.0 + 0.0
    
    f_expanded = torch.repeat_interleave(f, S)
    buf = s * f_expanded
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final normalization (no per-block factor division, just world_size)
    acc = acc / world_size
    
    return acc