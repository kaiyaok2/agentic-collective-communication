
def tenariindep_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Instead of 10 separate all_reduce operations on the same input,
    # do one all_reduce and multiply by 10
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 10.0 * reduced
