
def evolved_p4803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 1: Remove dead call
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    return 3 * ax
