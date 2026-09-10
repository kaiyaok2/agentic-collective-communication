
def evolved_p6200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return 6 * xm.all_reduce(xm.REDUCE_SUM, x)
