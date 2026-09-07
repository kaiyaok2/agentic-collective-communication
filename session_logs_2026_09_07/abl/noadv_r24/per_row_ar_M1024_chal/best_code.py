
def evolved_p7200(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR full: AllReduce SUM on the entire tensor
    # Input x: (1024, 64), Output: (1024, 64) with element-wise sum across all ranks
    return xm.all_reduce(xm.REDUCE_SUM, x)
