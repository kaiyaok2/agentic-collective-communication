def eightyaltsum_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # The sum 1 - 2 + 3 - 4 + ... + 79 - 80 = -40
    # So we can do one all_reduce and multiply by -40
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return -40.0 * reduced