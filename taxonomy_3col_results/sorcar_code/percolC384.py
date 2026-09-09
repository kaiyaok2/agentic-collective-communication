
def percolC384_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on entire tensor reduces all elements across ranks
    # This is equivalent to reducing each column independently but much more efficient
    return xm.all_reduce(xm.REDUCE_SUM, x)
