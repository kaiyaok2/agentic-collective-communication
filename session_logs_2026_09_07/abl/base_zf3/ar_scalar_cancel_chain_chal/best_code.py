
def evolved_p6103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # y = 4 * AR(x) where AR is all-reduce SUM
    # Single all-reduce followed by scalar multiplication
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    y = 4 * ax
    return y
