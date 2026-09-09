
def evolved_p7100(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce MAX on full tensor (512, 16)
    # Returns element-wise max across all ranks
    return xm.all_reduce(xm.REDUCE_MAX, x)
