
def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape to separate batches for vectorized operations
    s_reshaped = s.view(B, S)
    
    for _ in range(6):
        # Compute factors for all batches at once
        f = 1.0 + (s_reshaped * s_reshaped).mean(dim=1, keepdim=True)
        
        # Apply factors
        buf = (s_reshaped * f).view(-1)
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize
        acc_reshaped = acc.view(B, S)
        s_reshaped = acc_reshaped / (world_size * f)
    
    # Last iteration
    f = 1.0 + (s_reshaped * s_reshaped).mean(dim=1, keepdim=True)
    buf = (s_reshaped * f).view(-1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    return acc
