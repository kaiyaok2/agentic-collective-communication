
def thirtytwoalt_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Alternating sum: 1 - 2 + 3 - 4 + ... + 31 - 32 = -16
    # Factor out the all_reduce operation
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return -16.0 * reduced
