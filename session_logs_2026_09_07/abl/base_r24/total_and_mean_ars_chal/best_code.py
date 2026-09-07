def evolved_p6701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute all_reduce once
    total = xm.all_reduce(xm.REDUCE_SUM, x)
    # Reuse for mean
    mean = total / world_size
    return total + mean