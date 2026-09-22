def r56_hier_d4_n1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    NG = 4
    D = 4
    
    # Create groups: partition ranks by rank mod 4
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    
    # Compute rank-specific scaling factors
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Initialize current tensor
    cur = x
    
    # Four-stage alternating dispatch
    # Stage 0 (t=0, even): group-wise all_reduce with a[r] scaling
    cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
    
    # Stage 1 (t=1, odd): global all_reduce with b[r] scaling
    cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    # Stage 2 (t=2, even): group-wise all_reduce with a[r] scaling
    cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
    
    # Stage 3 (t=3, odd): global all_reduce with b[r] scaling
    cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    
    return cur