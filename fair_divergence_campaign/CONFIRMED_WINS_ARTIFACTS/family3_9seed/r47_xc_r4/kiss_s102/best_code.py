def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # First all_reduce sums across all ranks
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    # Each subsequent iteration multiplies by world_size, so multiply by world_size^3
    result = result * (world_size ** 3)
    return result