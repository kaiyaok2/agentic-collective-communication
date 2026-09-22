def r57_a2asum_d2_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute all scaling constants
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Precompute combined scaling factor
    scale_factor = a / u
    
    # Stage 0: First iteration
    # Fuse initial scaling with first all_reduce
    y0 = a * x
    z0 = xm.all_reduce(xm.REDUCE_SUM, y0)
    
    # Stage 1: Second iteration  
    # Fuse the division and multiplication: (z0 / u) * a = z0 * (a / u)
    y1 = scale_factor * z0
    z1 = xm.all_reduce(xm.REDUCE_SUM, y1)
    
    # Final scaling
    result = z1 / u
    
    return result