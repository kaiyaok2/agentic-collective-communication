def evolved_p6102(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce sum across all ranks
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    # Apply ReLU activation: max(0, x)
    return ax.clamp(min=0)