
def thirtytwoalt_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Original: 1*sum(x) - 2*sum(x) + 3*sum(x) - ... - 32*sum(x)
    # = sum(x) * (1 - 2 + 3 - 4 + ... + 31 - 32) = sum(x) * (-16)
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return -16.0 * reduced
