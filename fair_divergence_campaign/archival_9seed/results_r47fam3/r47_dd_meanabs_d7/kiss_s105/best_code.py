
def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_view = s.view(B, S)
    
    for iteration in range(6):
        # Compute all scaling factors at once
        f = (1.0 + s_view.mean(dim=1).abs()).unsqueeze(1)
        
        # Apply scaling using broadcasting
        buf = (s_view * f).view(-1)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 5:
            # Apply inverse scaling
            s_view = (acc.view(B, S) / (world_size * f))
        else:
            # Last iteration - just divide by world_size
            s_view = (acc / world_size).view(B, S)
    
    return s_view.view(-1)
