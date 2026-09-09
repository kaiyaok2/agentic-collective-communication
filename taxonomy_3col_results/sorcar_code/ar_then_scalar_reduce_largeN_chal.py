
def evolved_p5801(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Sum locally first to reduce to scalar, then all_reduce the scalar
    local_sum = x.sum()  # Reduce tensor to scalar locally
    global_sum = xm.all_reduce(xm.REDUCE_SUM, local_sum)  # All-reduce just the scalar
    return global_sum
