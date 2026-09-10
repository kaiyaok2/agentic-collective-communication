
def evolved_p6200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Optimize: a + b + c = all_reduce(x) + 2*all_reduce(x) + 3*all_reduce(x)
    #                     = 6 * all_reduce(x)
    # Reduces from 3 all_reduce ops to just 1
    result = 6 * xm.all_reduce(xm.REDUCE_SUM, x)
    return result
