
def evolved_p6203(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine all operations into a single all_reduce
    # a + b + c = all_reduce(x) + all_reduce(2*x) + all_reduce(2*x)
    #           = all_reduce(5*x)
    result = xm.all_reduce(xm.REDUCE_SUM, 5 * x)
    return result
