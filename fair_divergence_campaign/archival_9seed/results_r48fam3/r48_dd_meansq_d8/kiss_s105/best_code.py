
def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(7):
        # Compute all factors at once using view
        s_blocks = s.view(B, S)
        f_values = 1.0 + (s_blocks * s_blocks).mean(dim=1)
        
        # Create factor tensor using unsqueeze and expand
        f_tensor = f_values.unsqueeze(1).expand(B, S).reshape(B * S)
        
        # Vectorized operations
        buf = s * f_tensor
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 6:
            acc = acc / (world_size * f_tensor)
        else:
            acc = acc / world_size
        
        s = acc
    
    return s
