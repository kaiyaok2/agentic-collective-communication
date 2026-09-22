def r56_hier_d8_n1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Naive Sequential Alternating AllReduce strategy.
    
    Implements 8 sequential all_reduce calls alternating between:
    - Even iterations (t=0,2,4,6): group-wise all_reduce with scalar a[r]
    - Odd iterations (t=1,3,5,7): global all_reduce with scalar b[r]
    
    This is the baseline implementation with maximal dispatches (8 total),
    minimal fusion, and straightforward correctness.
    """
    W = world_size
    NG = 4  # Number of groups
    D = 8   # Depth (number of iterations)
    
    # Create groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-specific scalar multipliers
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Initialize current value
    cur = x
    
    # Execute 8 sequential all_reduce operations
    for t in range(D):
        if t % 2 == 0:
            # Even iteration: group-wise all_reduce with scalar a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd iteration: global all_reduce with scalar b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur