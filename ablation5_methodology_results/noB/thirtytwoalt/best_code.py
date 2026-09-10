
def thirtytwoalt_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # The original implementation computes:
    # 1.0 * all_reduce(x) - 2.0 * all_reduce(x) + 3.0 * all_reduce(x) - ... - 32.0 * all_reduce(x)
    # Which simplifies to: all_reduce(x) * (1 - 2 + 3 - 4 + ... + 31 - 32)
    # The sum is: (1-2) + (3-4) + ... + (31-32) = -1 * 16 = -16
    return -16.0 * xm.all_reduce(xm.REDUCE_SUM, x)
