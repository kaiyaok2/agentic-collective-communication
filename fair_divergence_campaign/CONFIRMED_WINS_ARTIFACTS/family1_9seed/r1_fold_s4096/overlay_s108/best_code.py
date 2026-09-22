def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                     cores_per_device, xm, torch, num_nodes=1):
    """
    Streamlined implementation that minimizes operation count and dispatch overhead.
    Reduces the number of separate tensor operations between collectives.
    """
    S = 4096
    W = world_size
    dtype = x.dtype
    
    # Precompute all scaling factors as a single tensor operation
    a = 1.0 + 0.5 * (torch.arange(W, dtype=dtype, device=x.device) % 3)
    
    # First all-reduce: sum all inputs
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse reshape and first scaling into minimal operations
    # Scale by a[r]/W in-place style computation
    s1 = s1.view(W, S) * (a / W).view(W, 1)
    
    # Second all-reduce on flattened result
    s1 = xm.all_reduce(xm.REDUCE_SUM, s1.view(-1))
    
    # Fuse reshape and second scaling (simplified to 1/W)
    # Since we divide by a[r] then multiply by a[r], they cancel
    s1 = s1.view(W, S) * (1.0 / W)
    
    # Third all-reduce: sum the final scaled values
    out = xm.all_reduce(xm.REDUCE_SUM, s1.view(-1))
    
    return out