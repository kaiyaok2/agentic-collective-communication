def r56_hier_d6_n1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential Alternating All-Reduce Chain
    
    6 separate all_reduce dispatches alternating between group-scoped (even stages)
    and global (odd stages), each scaling the input by rank-dependent coefficients.
    """
    W = world_size
    NG = 4  # Number of groups
    D = 6   # Depth
    
    # Create groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-dependent coefficients
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Initialize current value
    cur = x
    
    # Execute 6 alternating all-reduce stages
    for t in range(D):
        if t % 2 == 0:
            # Even stages (0, 2, 4): group-scoped all_reduce with coefficient a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd stages (1, 3, 5): global all_reduce with coefficient b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur