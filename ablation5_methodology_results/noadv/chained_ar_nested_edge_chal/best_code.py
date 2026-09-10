def evolved_p3701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax1 = xm.all_reduce(xm.REDUCE_SUM, x)
    y1 = ax1 * 2
    ax2 = xm.all_reduce(xm.REDUCE_SUM, x + y1)
    y2 = ax2 + 3
    ax3 = xm.all_reduce(xm.REDUCE_SUM, x + y2)
    return ax3
