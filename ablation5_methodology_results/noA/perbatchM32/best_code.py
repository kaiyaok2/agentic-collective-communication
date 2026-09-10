
def perbatchM32_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on entire tensor instead of per-batch
    # all_reduce operates element-wise, so this is semantically equivalent
    # but much more efficient (1 collective instead of x.shape[0] collectives)
    return xm.all_reduce(xm.REDUCE_SUM, x)
