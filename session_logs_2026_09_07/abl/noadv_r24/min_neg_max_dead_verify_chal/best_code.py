
def evolved_p6301(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: element-wise MIN across all ranks
    # Direct approach: single REDUCE_MIN all-reduce
    return xm.all_reduce(xm.REDUCE_MIN, x)
