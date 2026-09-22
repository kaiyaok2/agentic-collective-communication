def r56_hier_d4_n1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential Four-Stage Dispatch strategy.
    
    Performs 4 alternating all_reduce operations:
    - Even stages (0, 2): group-wise all_reduce with scalar a
    - Odd stages (1, 3): global all_reduce with scalar b
    
    This is a direct translation of the reference implementation,
    using 4 separate xm.all_reduce calls for maximum clarity.
    """
    W = world_size
    NG = 4  # Number of groups
    D = 4   # Depth (number of stages)
    
    # Create groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-specific scalars
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Initialize current result
    cur = x
    
    # Execute 4 stages sequentially
    for t in range(D):
        if t % 2 == 0:
            # Even stages: group-wise all_reduce with scalar a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd stages: global all_reduce with scalar b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur