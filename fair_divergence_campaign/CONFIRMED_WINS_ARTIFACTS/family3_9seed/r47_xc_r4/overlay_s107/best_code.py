def r47_xc_r4_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential reduce-scatter all-gather chain strategy.
    
    Performs 4 consecutive reduce_scatter+all_gather pairs with no fusion.
    This is the baseline implementation with 8 total dispatches (4 RS + 4 AG).
    
    Each round:
    1. reduce_scatter: sum-reduces and scatters across world_size
    2. all_gather: gathers the scattered results back to full size
    
    The final result is the elementwise sum of x across all ranks, scaled by world_size^3.
    """
    s = x
    
    # Round 1: reduce_scatter + all_gather
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0, shard_count=world_size)
    rs = rs * 1.0
    s = xm.all_gather(rs, dim=0)
    
    # Round 2: reduce_scatter + all_gather
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0, shard_count=world_size)
    rs = rs * 1.0
    s = xm.all_gather(rs, dim=0)
    
    # Round 3: reduce_scatter + all_gather
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0, shard_count=world_size)
    rs = rs * 1.0
    s = xm.all_gather(rs, dim=0)
    
    # Round 4: reduce_scatter + all_gather
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0, shard_count=world_size)
    rs = rs * 1.0
    s = xm.all_gather(rs, dim=0)
    
    return s