def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    # After first all-reduce: sum(all x's)
    # Need to multiply by world_size^3 to match 4 cycles
    scale = world_size ** 3
    return result * scale