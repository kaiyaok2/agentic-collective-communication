
def thirtytwoalt_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Original computes: 1*all_reduce - 2*all_reduce + 3*all_reduce - ... - 32*all_reduce
    # This simplifies to: (1-2+3-4+...+31-32) * all_reduce(x) = -16 * all_reduce(x)
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return -16.0 * result
