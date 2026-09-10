
def evolved_p6200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return 6 * result
