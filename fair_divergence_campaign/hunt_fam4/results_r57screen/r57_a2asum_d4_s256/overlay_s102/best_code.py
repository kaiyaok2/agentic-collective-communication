def r57_a2asum_d4_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 4
    
    # Compute scaling factor for this rank
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    
    # Compute normalization constant
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Pre-compute the combined scaling factor
    scale_factor = a / u
    
    # Reshape once at the beginning
    cur = x.reshape(W, S)
    
    for _t in range(D):
        # Scale and all-reduce in one conceptual step
        # First scale locally
        cur = scale_factor * cur
        
        # All-reduce sums across ranks element-wise
        cur = xm.all_reduce(xm.REDUCE_SUM, cur)
    
    # Flatten back to (W*S,) at the end
    return cur.reshape(W * S)