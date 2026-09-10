
def evolved_p5102(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # First perform local reduction along dimension 0
    local_sum = x.sum(dim=0)  # shape (N,) - reduces M*N to N elements
    # Then all-reduce the smaller tensor
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)  # transfers only N elements
    return result
