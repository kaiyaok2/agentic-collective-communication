def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Hierarchical two-level reduction pattern.
    
    Strategy: Group the four rounds into two meta-rounds:
    - First meta-round: 2 RS+AG cycles
    - Second meta-round: Apply 2 RS+AG cycles again on the result
    
    This creates a hierarchical pattern that can exploit node-local bandwidth
    when world_size factorizes nicely (e.g., multi-node settings).
    
    Total: 8 dispatches (4 reduce_scatter + 4 all_gather)
    """
    s = x
    
    # First meta-round: 2 RS+AG cycles
    # Cycle 1
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    s = xm.all_gather(rs, dim=0)
    
    # Cycle 2
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    s = xm.all_gather(rs, dim=0)
    
    # Second meta-round: 2 RS+AG cycles on the accumulated result
    # Cycle 3
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    s = xm.all_gather(rs, dim=0)
    
    # Cycle 4
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    s = xm.all_gather(rs, dim=0)
    
    return s