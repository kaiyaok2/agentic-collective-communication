
def evolved_p7200(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 2: Single all-reduce on full tensor
    return xm.all_reduce(xm.REDUCE_SUM, x)
