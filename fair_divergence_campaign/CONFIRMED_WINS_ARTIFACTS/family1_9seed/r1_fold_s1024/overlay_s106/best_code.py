def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Compute scale factors ahead of time and combine more operations
    scale1 = torch.tensor([a[r] / W for r in range(W)], dtype=dtype, device=x.device).view(-1, 1).expand(W, S).reshape(-1)
    scale2_combined = torch.tensor([(1.0 / max(a[r], 1e-9)) * (a[r] / W) for r in range(W)], dtype=dtype, device=x.device).view(-1, 1).expand(W, S).reshape(-1)
    
    # First all-reduce with immediate scale operation
    buf = xm.all_reduce(xm.REDUCE_SUM, x) * scale1
    
    # Second all-reduce with immediate scale operation
    buf = xm.all_reduce(xm.REDUCE_SUM, buf) * scale2_combined
    
    # Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return out