def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    # Initial all_reduce and reshape to (B, S)
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # Perform 7 iterations
    for iteration in range(7):
        # Compute scaling factors and apply
        f = 1.0 + 3.0 * (s * s).mean(dim=1, keepdim=True)
        
        # All-reduce sum (flattening inline)
        acc = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1))
        
        # Normalize and prepare for next iteration
        if iteration < 6:
            s = acc.view(B, S) / (world_size * f)
        else:
            return acc / world_size
    
    return s.view(-1)