
def evolved_p6301(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute elementwise MIN across all ranks
    # Single all-reduce with REDUCE_MIN operation
    return xm.all_reduce(xm.REDUCE_MIN, x)
