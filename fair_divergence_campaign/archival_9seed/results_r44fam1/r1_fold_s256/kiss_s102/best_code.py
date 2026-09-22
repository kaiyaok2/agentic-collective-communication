
def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Create multiplier using arange and modular arithmetic
    # a[r] = 1.0 + 0.5*(r % 3), so pattern is [1.0, 1.5, 2.0, 1.0, 1.5, 2.0, ...]
    indices = torch.arange(W * S, device=x.device, dtype=x.dtype)
    r_indices = (indices // S) % 3  # Maps to 0,1,2,0,1,2...
    mult1 = (1.0 + 0.5 * r_indices) / W
    
    # First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply first transformation
    buf0 = s1 * mult1
    
    # Second all-reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Combined transformation: 1/W
    buf1 = s2 / W
    
    # Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, buf1)
    return out
