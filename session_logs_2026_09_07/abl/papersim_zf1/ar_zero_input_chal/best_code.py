
def evolved_p4803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: return 3 * AR(x)
    # where AR is all_reduce with SUM operation
    # Simply all-reduce x and multiply by 3
    return 3 * xm.all_reduce(xm.REDUCE_SUM, x)
