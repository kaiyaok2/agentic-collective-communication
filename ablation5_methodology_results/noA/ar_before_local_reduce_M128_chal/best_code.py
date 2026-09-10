
def evolved_p5102(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # First sum locally along dimension 0 (M x N -> N)
    local_sum = x.sum(dim=0)
    # Then all_reduce the N-element result
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)
