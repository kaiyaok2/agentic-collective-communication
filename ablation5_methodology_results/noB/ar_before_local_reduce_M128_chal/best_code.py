def evolved_p5102(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Sum locally first to reduce data size
    local_sum = x.sum(dim=0)
    # Then all_reduce the smaller tensor
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result