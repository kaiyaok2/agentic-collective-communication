def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors as a tensor (vectorized)
    a = torch.tensor([1.0 + 0.5*(r % 3) for r in range(W)], dtype=dtype)
    a_expanded = a.repeat_interleave(S)  # Shape: [W*S]
    
    # Stage 1: First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stage 2: Scale and all-reduce (vectorized)
    s = (a_expanded * s) / W
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    # Stage 3: Unscale, scale again, and all-reduce (vectorized)
    s = s / a_expanded.clamp(min=1e-9)
    s = (a_expanded * s) / W
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    # Stage 4: Final unscale, scale, and all-reduce (vectorized)
    s = s / a_expanded.clamp(min=1e-9)
    s = (a_expanded * s) / W
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s