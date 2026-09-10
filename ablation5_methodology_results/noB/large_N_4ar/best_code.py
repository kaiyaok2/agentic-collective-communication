def large_N_4ar_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return 4.0 * xm.all_reduce(xm.REDUCE_SUM, x)