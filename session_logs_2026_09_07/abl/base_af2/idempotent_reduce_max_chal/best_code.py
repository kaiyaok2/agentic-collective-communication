
def evolved_p5103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute elementwise max across all ranks - single operation
    return xm.all_reduce(xm.REDUCE_MAX, x)
