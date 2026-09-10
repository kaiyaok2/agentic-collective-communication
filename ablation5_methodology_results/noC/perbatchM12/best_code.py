
def perbatchM12_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on entire tensor instead of per-batch reduces
    return xm.all_reduce(xm.REDUCE_SUM, x)
