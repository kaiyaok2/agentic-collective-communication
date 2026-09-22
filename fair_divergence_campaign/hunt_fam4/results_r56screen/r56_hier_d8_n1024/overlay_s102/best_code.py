def r56_hier_d8_n1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential alternating all-reduce loop strategy.
    
    Direct translation of reference with 8 separate xm.all_reduce dispatches 
    alternating between group-scoped (even iterations) and global (odd iterations).
    Each dispatch waits for the previous to complete.
    """
    W = world_size
    NG = 4  # Number of groups
    D = 8   # Depth
    
    # Create groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-specific scaling factors
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Initialize current tensor
    cur = x
    
    # Execute 8 sequential all-reduce operations
    for t in range(D):
        if t % 2 == 0:
            # Even iterations: group-scoped all-reduce with factor a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd iterations: global all-reduce with factor b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur