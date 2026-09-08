
def evolved_p5701(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: 2 * AR(x)
    # Compute all-reduce once and multiply by 2
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 2 * reduced
