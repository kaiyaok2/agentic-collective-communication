
def evolved_p6900(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR full = AllReduce SUM on the entire tensor
    # Input: x is (96, 512)
    # Output: same shape, each element summed across all ranks
    return xm.all_reduce(xm.REDUCE_SUM, x)
