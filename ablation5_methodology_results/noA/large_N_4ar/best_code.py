def large_N_4ar_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    r1 = xm.all_reduce(xm.REDUCE_SUM, x)
    r2 = xm.all_reduce(xm.REDUCE_SUM, x)
    r3 = xm.all_reduce(xm.REDUCE_SUM, x)
    r4 = xm.all_reduce(xm.REDUCE_SUM, x)
    return r1 + r2 + r3 + r4