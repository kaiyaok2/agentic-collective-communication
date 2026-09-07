
def evolved_p6200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute y = AR(x) + 2*AR(x) + 3*AR(x) = 6*AR(x)
    # Call all_reduce once instead of three times
    ar = xm.all_reduce(xm.REDUCE_SUM, x)
    return 6 * ar
