def r47_xc_r4_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential reduce-scatter all-gather chain.
    Four consecutive reduce_scatter+all_gather pairs executed sequentially.
    8 total collective dispatches (4 RS + 4 AG).
    """
    s = x
    
    # Round 1: reduce_scatter + all_gather
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    rs = rs * 1.0
    s = xm.all_gather(rs, dim=0)
    
    # Round 2: reduce_scatter + all_gather
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    rs = rs * 1.0
    s = xm.all_gather(rs, dim=0)
    
    # Round 3: reduce_scatter + all_gather
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    rs = rs * 1.0
    s = xm.all_gather(rs, dim=0)
    
    # Round 4: reduce_scatter + all_gather
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    rs = rs * 1.0
    s = xm.all_gather(rs, dim=0)
    
    return s