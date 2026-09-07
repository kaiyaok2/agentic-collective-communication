
def evolved_p4800(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: a = 2 * AR(x); b = 3 * AR(x); return a + b
    # Simplifies to: (2 + 3) * AR(x) = 5 * AR(x)
    # Single all-reduce instead of two
    ar_x = xm.all_reduce(xm.REDUCE_SUM, x)
    return 5 * ar_x
