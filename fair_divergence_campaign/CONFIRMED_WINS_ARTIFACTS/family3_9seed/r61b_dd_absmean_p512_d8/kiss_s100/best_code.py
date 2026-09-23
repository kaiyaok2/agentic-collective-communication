
def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_2d = s.view(B, S)
    
    for iteration in range(7):
        # Compute factors for all batches at once
        factors = 1.0 + s_2d.abs().mean(dim=1)  # Shape: (B,)
        
        # Multiply each batch by its factor using broadcasting
        buf_2d = s_2d * factors.unsqueeze(1)
        
        # All-reduce (need 1D tensor)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf_2d.view(-1))
        acc_2d = acc.view(B, S)
        
        if iteration < 6:
            # Divide by world_size * factors
            s_2d = acc_2d / (world_size * factors.unsqueeze(1))
        else:
            # Last iteration: just divide by world_size
            s_2d = acc_2d / world_size
    
    return s_2d.view(-1)
