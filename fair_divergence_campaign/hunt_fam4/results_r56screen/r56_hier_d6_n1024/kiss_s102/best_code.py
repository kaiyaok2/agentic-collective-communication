def r56_hier_d6_n1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; NG = 4; D = 6
    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0
    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0
    cur = x
    for t in range(D):
        if t % 2 == 0:
            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)
        else:
            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)
    return cur
