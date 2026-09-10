
def twentyeightinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return 28.0 * result
