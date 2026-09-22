def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                     cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    dtype = x.dtype
    
    # Pre-compute all scale coefficients at once
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], dtype=dtype, device=x.device)
    
    # Pre-compute scales to avoid redundant computation
    scales_1 = a / W
    
    # Stage 1: all_reduce(x) and apply scale in one step
    buf = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = buf.view(W, S) * scales_1.view(W, 1)
    buf = buf.view(-1)
    
    # Stage 2: all_reduce and apply scale
    buf = xm.all_reduce(xm.REDUCE_SUM, buf)
    inv_W = 1.0 / W
    buf = buf * inv_W
    
    # Stage 3: all_reduce (final)
    out = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return out