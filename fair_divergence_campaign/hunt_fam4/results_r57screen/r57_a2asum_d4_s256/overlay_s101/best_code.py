def r57_a2asum_d4_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 4
    
    # Compute scaling factors
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Preserve input dtype
    dtype = x.dtype
    
    cur = x
    
    for _t in range(D):
        # Fused scale operation
        scaled = a * cur
        
        # reduce_scatter: each rank gets sum of its corresponding S-sized block from all ranks
        # Input: (W*S,) scaled tensor
        # Output: (S,) tensor (sum of S-sized blocks from all ranks)
        z = xm.reduce_scatter(
            xm.REDUCE_SUM,
            scaled,
            scale=1.0,
            scatter_dim=0,
            shard_count=W,
            groups=None
        )
        
        # all_gather: gather all shards to reconstruct full tensor
        cur = xm.all_gather(z, dim=0)
        
        # Scale by 1/u
        cur = cur / u
    
    return cur