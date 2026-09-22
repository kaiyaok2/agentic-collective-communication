def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                     cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    a = torch.tensor([1.0 + 0.5*(r % 3) for r in range(W)], device=x.device)
    
    # First all-reduce
    buf = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse all local scaling: (a[r] * buf / W) / max(a[r], 1e-9) simplifies to buf / (W * max(a[r], 1e-9))
    # But max(a[r], 1e-9) ≈ a[r] for our values, so it's buf / W
    buf = buf / W
    
    # Second all-reduce
    buf = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Fuse remaining scaling operations with vectorization
    # Create scaling vector for all positions at once
    scale_vec = (a / W).repeat_interleave(S)
    buf = buf * scale_vec
    
    # Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return out