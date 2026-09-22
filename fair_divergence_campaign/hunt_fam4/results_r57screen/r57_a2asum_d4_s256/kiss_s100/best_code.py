
def r57_a2asum_d4_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 4
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    A_tot = sum(1.0 + 0.5*((k*11) % 7)/7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Combine scaling factors to reduce operations
    scale_factor = a / u
    
    cur = x
    for _t in range(D):
        z = xm.reduce_scatter(xm.REDUCE_SUM, scale_factor * cur, scale=1.0, 
                            scatter_dim=0, shard_count=W)
        cur = xm.all_gather(z, dim=0)
    
    return cur
