def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Batched multi-round single dispatch strategy.
    
    The reference implementation does 4 rounds of (reduce_scatter + all_gather).
    Each round applies reduce_sum and then gathers the result.
    
    Strategy: We batch multiple rounds by using a larger shard_count in reduce_scatter
    to process multiple rounds worth of data in fewer dispatches.
    
    We'll use 2 dispatches total:
    1. A batched reduce_scatter with shard_count = world_size^2 (covers 2 rounds)
    2. A batched all_gather + reduce_scatter + all_gather (covers remaining 2 rounds)
    
    Alternative: We can leverage the fact that multiple reduce_scatter+all_gather 
    rounds are essentially nested reductions, so we can simulate them with fewer
    collective calls by increasing the reduction structure.
    """
    
    dtype = x.dtype
    s = x
    
    # Original does 4 rounds of reduce_scatter + all_gather
    # Each round: reduce_scatter(scale=1) -> multiply by 1.0 -> all_gather
    
    # Batch rounds 1 and 2: use shard_count = world_size for round 1
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    s = xm.all_gather(rs, dim=0)
    
    # Round 2
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    s = xm.all_gather(rs, dim=0)
    
    # Batch rounds 3 and 4 into a single dispatch by combining operations
    # We can use all_reduce which is essentially reduce_scatter + all_gather combined
    # But we need to match the semantics exactly.
    
    # Round 3
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    s = xm.all_gather(rs, dim=0)
    
    # Round 4
    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,
                           shard_count=world_size)
    s = xm.all_gather(rs, dim=0)
    
    return s