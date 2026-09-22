def r56_hier_d8_n4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Naive sequential alternating dispatch strategy:
    Perform 8 separate all_reduce dispatches in a loop, alternating between 
    group-wise (with scalar multiply a[r]) and global (with scalar multiply b[r]) operations.
    """
    W = world_size
    NG = 4
    D = 8
    
    # Build groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-specific scalars
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Initialize current result
    cur = x
    
    # Perform 8 alternating dispatches
    for t in range(D):
        if t % 2 == 0:
            # Even stages (0, 2, 4, 6): group-wise all_reduce with scalar a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd stages (1, 3, 5, 7): global all_reduce with scalar b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur