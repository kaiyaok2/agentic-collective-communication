def evolved_p5102(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # First do local sum across dim 0: (M, N) -> (N,)
    local_sum = x.sum(dim=0)
    # Then all-reduce the smaller vector
    global_sum = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return global_sum