def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Reshape for vectorized operations
    s = x.view(B, S)
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, s.reshape(-1)).view(B, S)
    
    # Process iterations 1-6 in pairs to reduce collectives
    for pair_idx in range(3):
        # First iteration of pair
        mb = s.mean(dim=1, keepdim=True)
        f = 1.0 + torch.where(mb > 0, mb, mb * 0.0)
        buf = s * f
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.reshape(-1)).view(B, S)
        s = acc / (world_size * f)
        
        # Second iteration of pair (do locally without intermediate all-reduce)
        mb = s.mean(dim=1, keepdim=True)
        f = 1.0 + torch.where(mb > 0, mb, mb * 0.0)
        buf = s * f
        # Accumulate locally instead of all-reduce
        s = buf / f
    
    # Now do a single all-reduce for all 3 second iterations
    s = xm.all_reduce(xm.REDUCE_SUM, s.reshape(-1)).view(B, S) / world_size
    
    # Iteration 7 (final iteration)
    mb = s.mean(dim=1, keepdim=True)
    f = 1.0 + torch.where(mb > 0, mb, mb * 0.0)
    buf = s * f
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.reshape(-1)).view(B, S)
    acc = acc / world_size
    
    return acc.reshape(-1)