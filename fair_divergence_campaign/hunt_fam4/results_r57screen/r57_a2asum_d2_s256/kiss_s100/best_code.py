
def r57_a2asum_d2_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 2
    
    # Compute scaling factor for this rank
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    
    # Compute total scaling factor
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Pre-compute combined scaling factor
    scale_factor = a / u
    
    cur = x
    
    for _t in range(D):
        # Combined scale and reduce
        cur = xm.all_reduce(xm.REDUCE_SUM, scale_factor * cur)
    
    return cur
