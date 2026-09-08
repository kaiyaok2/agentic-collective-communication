
def evolved_p4803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: return 3 * all_reduce_sum(x)
    # Eliminate the dead all_reduce(zeros) from baseline
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    return 3 * ax
