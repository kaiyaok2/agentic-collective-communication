
def evolved_p6600(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce the entire tensor at once instead of 64 separate calls
    # Each element x[i,j] becomes sum of x[i,j] across all ranks
    return xm.all_reduce(xm.REDUCE_SUM, x)
