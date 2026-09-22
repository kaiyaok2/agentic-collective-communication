def r2_deep4_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Compute per-rank scaling factors as a tensor (vectorized)
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], dtype=x.dtype, device=x.device)
    
    # Reshape for broadcasting: (W, 1) so we can multiply (W, S) shaped slices
    a_expanded = a.view(W, 1).expand(W, S).reshape(-1)
    a_clamped = torch.clamp(a_expanded, min=1e-9)
    W_inv = 1.0 / W
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Stage 1: All-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse stages 1-2: Local scaling, all-reduce, and prepare for next stage
    buf = (a_expanded * s) * W_inv
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Fuse stages 2-3: Unscale, scale, all-reduce
    buf = (s / a_clamped) * a_expanded * W_inv
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Fuse stages 3-4: Unscale, scale, all-reduce (final)
    buf = (s / a_clamped) * a_expanded * W_inv
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s