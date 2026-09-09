
def percolC512_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on entire tensor instead of per-column
    # This computes element-wise sum across all ranks
    return xm.all_reduce(xm.REDUCE_SUM, x)
