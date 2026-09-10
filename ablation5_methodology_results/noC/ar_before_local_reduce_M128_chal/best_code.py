def evolved_p5102(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    local_sum = x.sum(dim=0)  # Sum along dimension 0 locally first
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)  # All-reduce the smaller tensor
    return result