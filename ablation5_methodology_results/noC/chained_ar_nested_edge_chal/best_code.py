
def evolved_p3701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax1 = xm.all_reduce(xm.REDUCE_SUM, x)
    coeff = 1 + world_size + 2 * world_size * world_size
    return ax1 * coeff + 3 * world_size
