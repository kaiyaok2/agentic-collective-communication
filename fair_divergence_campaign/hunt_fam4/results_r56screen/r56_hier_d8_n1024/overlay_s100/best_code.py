def r56_hier_d8_n1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Naive Sequential Eight-Stage Loop implementation.
    
    Direct translation of reference: 8 sequential xm.all_reduce dispatches 
    alternating between group-wise (even stages) and global (odd stages), 
    applying scalar multipliers a or b before each collective.
    
    - Even stages (t=0,2,4,6): group-wise all_reduce with multiplier a
    - Odd stages (t=1,3,5,7): global all_reduce with multiplier b
    """
    W = world_size
    NG = 4  # Number of groups
    D = 8   # Depth/number of stages
    
    # Create groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-specific scalar multipliers
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Initialize current result with input
    cur = x
    
    # Execute 8 sequential stages
    for t in range(D):
        if t % 2 == 0:
            # Even stage: group-wise all_reduce with multiplier a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd stage: global all_reduce with multiplier b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur