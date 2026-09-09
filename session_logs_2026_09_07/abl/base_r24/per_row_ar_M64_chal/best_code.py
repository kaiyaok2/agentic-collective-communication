
def evolved_p6600(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce the entire tensor in one collective call
    # instead of 64 separate row-wise all-reduces
    return xm.all_reduce(xm.REDUCE_SUM, x)
