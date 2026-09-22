def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Single all_reduce to sum across all ranks
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create scaling using repeat
    scale_base = torch.tensor(a, device=x.device, dtype=x.dtype).unsqueeze(1)
    scale = scale_base.repeat(1, S).view(-1)
    
    return s1 * scale