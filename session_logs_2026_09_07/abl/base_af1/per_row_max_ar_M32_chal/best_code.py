
def evolved_p6901(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR(MAX) full: All-Reduce with MAX operation on the entire tensor
    # Single all_reduce call on the full 2D tensor instead of 32 separate calls
    return xm.all_reduce(xm.REDUCE_MAX, x)
