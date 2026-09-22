def r56_hier_d8_n4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    NG = 4
    D = 8
    
    # Build groups: partition ranks by rank % 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-dependent scalars
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Initialize current result
    cur = x
    
    # 8 alternating all-reduce stages
    for t in range(D):
        if t % 2 == 0:
            # Even stages: group-scoped all_reduce with scalar a
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            # Odd stages: global all_reduce with scalar b
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur