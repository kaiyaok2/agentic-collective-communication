def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    scale = float(world_size ** 3)
    return s * scale