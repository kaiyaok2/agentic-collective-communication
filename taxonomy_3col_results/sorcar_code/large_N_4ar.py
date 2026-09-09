
def large_N_4ar_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    t = xm.all_reduce(xm.REDUCE_SUM, x)
    return 4.0 * t
