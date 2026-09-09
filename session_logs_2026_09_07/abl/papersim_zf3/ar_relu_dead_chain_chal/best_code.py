def evolved_p6102(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute y = relu(AR(x))
    # AR = all-reduce with SUM across all ranks
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    return ax.clamp(min=0)