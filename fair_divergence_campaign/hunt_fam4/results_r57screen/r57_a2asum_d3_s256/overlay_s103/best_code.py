def r57_a2asum_d3_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 3
    
    # Compute rank's scaling factor
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    
    # Compute total sum for normalization
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    dtype = x.dtype
    cur = x
    
    for _t in range(D):
        # Scale by rank's factor
        scaled = a * cur
        
        # Reduce-scatter: each rank gets sum of its corresponding block from all ranks
        # reduce_scatter divides input into W equal chunks, reduces corresponding chunks
        # across all ranks (with sum operation), and scatters results so each rank gets one
        # Input: (W*S,) tensor with W blocks of size S
        # Output: (S,) tensor - the sum of this rank's block from all ranks
        z = xm.reduce_scatter(
            xm.REDUCE_SUM,
            scaled,
            scale=1.0,
            scatter_dim=0,
            shard_count=W,
            groups=[]
        )
        
        # All-gather: collect all shards from all ranks
        # Input: (S,) tensor
        # Output: (W*S,) tensor
        gathered = xm.all_gather(z, dim=0)
        
        # Normalize
        cur = gathered / u
    
    return cur