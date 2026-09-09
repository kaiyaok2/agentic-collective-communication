
def evolved_p6102(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = relu(all_reduce_sum(x))
    # Step 1: All-reduce sum across all ranks
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    # Step 2: Apply relu (using clamp for compatibility)
    return ax.clamp(min=0)
