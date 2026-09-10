
def evolved_p4102(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine weighted inputs using linearity of all_reduce
    # Original: 2.5 * all_reduce(x) + 3.5 * all_reduce(y) + 7.5 * all_reduce(z)
    # Optimized: all_reduce(2.5 * x + 3.5 * y + 7.5 * z)
    combined = 2.5 * x + 3.5 * y + 7.5 * z
    s = xm.all_reduce(xm.REDUCE_SUM, combined)
    return s
