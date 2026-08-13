def evolved_p118(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return x - xm.all_reduce(xm.REDUCE_SUM, x) / world_size
