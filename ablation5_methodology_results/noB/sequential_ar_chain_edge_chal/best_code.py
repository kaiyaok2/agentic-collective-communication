def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    ax_p = ax * 2
    contrib = y + ax_p
    return xm.all_reduce(xm.REDUCE_SUM, contrib)
