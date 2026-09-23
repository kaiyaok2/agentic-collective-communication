def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape to process batches
    s_batched = s.view(B, S)
    
    # Compute factors for all batches at once
    factors = 1.0 + s_batched.abs().mean(dim=1)
    
    # Multiply each batch by its factor
    s_batched = s_batched * factors.unsqueeze(1)
    
    # Flatten back
    s = s_batched.view(-1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    s = s / world_size
    
    return s