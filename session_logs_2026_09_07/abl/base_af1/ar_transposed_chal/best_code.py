def evolved_p5401(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: AR(x) - all-reduce of x
    # Input: (128, 256), Output: (128, 256) identical on all ranks
    return xm.all_reduce(xm.REDUCE_SUM, x)