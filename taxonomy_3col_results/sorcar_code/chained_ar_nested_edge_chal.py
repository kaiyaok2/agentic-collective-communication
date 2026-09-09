
def evolved_p3701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax1 = xm.all_reduce(xm.REDUCE_SUM, x)
    # ax2 = ax1 * (1 + 2 * world_size)
    # ax3 = ax1 + ax2 * world_size + 3 * world_size
    # Expanding: ax3 = ax1 * (1 + world_size + 2 * world_size^2) + 3 * world_size
    return ax1 * (1 + world_size + 2 * world_size * world_size) + 3 * world_size
