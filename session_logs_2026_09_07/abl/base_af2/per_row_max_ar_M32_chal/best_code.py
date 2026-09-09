
def evolved_p6901(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR(MAX) full: all-reduce with MAX on entire tensor (32, 2048)
    # Single collective instead of 32 per-row reductions
    return xm.all_reduce(xm.REDUCE_MAX, x)
