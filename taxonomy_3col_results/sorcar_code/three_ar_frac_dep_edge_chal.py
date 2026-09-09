def evolved_p4102(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Apply coefficients before all_reduce (linearity property)
    weighted = 2.5 * x + 3.5 * y + 7.5 * z
    # Single all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, weighted)
    return s