def r56_hier_d4_n1024_fn(x, rank, world_size, num_devices,
                         cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential four-stage alternating all-reduce.
    
    Implements 4 stages alternating between:
    - Even stages (t=0,2): group-wise all-reduce with coefficient a
    - Odd stages (t=1,3): global all-reduce with coefficient b
    """
    W = world_size
    NG = 4  # Number of groups
    D = 4   # Depth (number of stages)
    
    # Create groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-dependent coefficients
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Start with input
    cur = x
    
    # Execute 4 stages
    for t in range(D):
        if t % 2 == 0:
            # Even stage: group-wise all-reduce with coefficient a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd stage: global all-reduce with coefficient b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur