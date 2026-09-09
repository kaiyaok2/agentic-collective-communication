
def evolved_p4803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: return 3 * AR(x)
    # where AR is all_reduce with REDUCE_SUM
    # Input x: (N,) local tensor
    # Output: (N,) identical on every rank
    
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    return 3 * ax
