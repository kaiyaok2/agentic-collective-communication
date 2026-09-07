
def evolved_p7001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: all-reduce (sum) the full tensor x across all ranks
    # Optimization: single all-reduce on entire tensor instead of 64 per-column calls
    return xm.all_reduce(xm.REDUCE_SUM, x)
