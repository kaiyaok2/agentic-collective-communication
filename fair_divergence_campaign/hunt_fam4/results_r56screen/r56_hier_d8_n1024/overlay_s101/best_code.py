def r56_hier_d8_n1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    NG = 4
    
    # Build groups
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute scaling factors
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Pre-compute compound scaling factors to reduce operations
    # Stage pattern: group(a), global(b), group(a), global(b), ...
    # We can combine consecutive scalings
    ab = a * b  # compound factor for group->global pairs
    
    # Execute in 4 pairs instead of 8 individual stages
    # Each pair: group(a) -> global(b) which compounds to ab
    
    cur = x
    
    # Pair 0: stages 0,1 - group(a) then global(b)
    cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
    cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    # Pair 1: stages 2,3 - group(a) then global(b)
    cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
    cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    # Pair 2: stages 4,5 - group(a) then global(b)
    cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
    cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    # Pair 3: stages 6,7 - group(a) then global(b)
    cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
    cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur