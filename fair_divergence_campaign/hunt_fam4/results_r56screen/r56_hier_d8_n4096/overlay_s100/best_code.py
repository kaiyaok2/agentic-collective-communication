def r56_hier_d8_n4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Naive sequential alternating collectives strategy:
    Direct translation of reference with 8 separate xm.all_reduce dispatches.
    Even iterations (t=0,2,4,6): group-scoped all_reduce with scaling factor a
    Odd iterations (t=1,3,5,7): global all_reduce with scaling factor b
    """
    W = world_size
    NG = 4
    D = 8
    
    # Create groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-specific scaling factors
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Initialize current tensor
    cur = x
    
    # 8 sequential all_reduce operations, alternating between group and global
    for t in range(D):
        if t % 2 == 0:
            # Even stage: group-scoped all_reduce with factor a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd stage: global all_reduce with factor b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur