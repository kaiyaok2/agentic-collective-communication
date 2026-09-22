def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    
    # Create weight pattern - try different arithmetic order
    r_idx = torch.arange(W, device=x.device, dtype=x.dtype)
    a_tensor = ((2.0 + (r_idx % 3)) / (2 * W * W)).view(W, 1).repeat(1, S).view(-1)
    
    # Chain operations
    return xm.all_reduce(xm.REDUCE_SUM, 
                        xm.all_reduce(xm.REDUCE_SUM, 
                                     a_tensor * xm.all_reduce(xm.REDUCE_SUM, x)))