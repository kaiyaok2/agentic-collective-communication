
def evolved_p4101(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first, then single all_reduce
    # This reduces 5 all_reduce ops to 1
    local_sum = x1 - 0.5 * x2 + 2.5 * x3 - 1.5 * x4 + 3.0 * x5
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
