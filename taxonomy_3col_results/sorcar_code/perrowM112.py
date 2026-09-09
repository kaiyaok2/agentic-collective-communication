
def perrowM112_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on the entire 2D tensor instead of per-row
    # This reduces 112 all_reduce operations to just 1
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return result
