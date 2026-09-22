def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_view = s.view(B, S)
    
    # First 5 iterations with per-block normalization
    for _ in range(5):
        # Compute factors for all blocks at once: (B,)
        f = s_view.mean(dim=1).abs() + 1.0
        
        # Scale all blocks using broadcasting: (B, S) * (B, 1)
        scaled = s_view * f.view(B, 1)
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, scaled.view(-1))
        
        # Normalize by world_size * f
        s_view = acc.view(B, S) / (world_size * f.view(B, 1))
    
    # 6th iteration - final normalization is different
    f = s_view.mean(dim=1).abs() + 1.0
    scaled = s_view * f.view(B, 1)
    acc = xm.all_reduce(xm.REDUCE_SUM, scaled.view(-1))
    
    # Final division by world_size only (no per-block f normalization)
    s = acc / world_size
    
    return s