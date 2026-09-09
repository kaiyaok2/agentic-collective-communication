
def evolved_p6600(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce the full tensor at once instead of 64 separate row all-reduces
    # Single collective operation eliminates 63 dispatch overheads
    return xm.all_reduce(xm.REDUCE_SUM, x)
