
def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with factor-based normalization
    for _ in range(6):
        # Reshape to compute per-batch statistics
        s_reshaped = s.view(B, S)
        f = 1.0 + s_reshaped.abs().mean(dim=1)  # shape: [B]
        f_expanded = f.repeat_interleave(S)  # shape: [B*S]
        
        # Multiply by factors
        buf = s * f_expanded
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by (world_size * factors)
        divisor = (world_size * f).repeat_interleave(S)
        s = acc / divisor
    
    # 7th iteration - final one with simpler normalization
    s_reshaped = s.view(B, S)
    f = 1.0 + s_reshaped.abs().mean(dim=1)
    f_expanded = f.repeat_interleave(S)
    
    buf = s * f_expanded
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / world_size
    
    return s
