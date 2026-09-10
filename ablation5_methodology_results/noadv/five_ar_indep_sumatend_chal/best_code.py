
def evolved_p5602(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Add all tensors locally first
    local_sum = x1 + x2 + x3 + x4 + x5
    # Then do one all_reduce on the combined sum
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
