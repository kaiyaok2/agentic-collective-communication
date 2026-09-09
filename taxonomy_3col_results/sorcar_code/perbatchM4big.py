
def perbatchM4big_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on entire tensor instead of per-batch
    # This reduces dispatch overhead from O(batch_size) to O(1)
    return xm.all_reduce(xm.REDUCE_SUM, x)
