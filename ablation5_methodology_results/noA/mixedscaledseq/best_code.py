def mixedscaledseq_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    acc = None
    t = 1.0 * xm.all_reduce(xm.REDUCE_SUM, x)
    acc = t if acc is None else acc + t
    t = -2.0 * xm.all_reduce(xm.REDUCE_SUM, x)
    acc = t if acc is None else acc + t
    t = 3.0 * xm.all_reduce(xm.REDUCE_SUM, x)
    acc = t if acc is None else acc + t
    t = -0.5 * xm.all_reduce(xm.REDUCE_SUM, x)
    acc = t if acc is None else acc + t
    t = 2.5 * xm.all_reduce(xm.REDUCE_SUM, x)
    acc = t if acc is None else acc + t
    return acc
