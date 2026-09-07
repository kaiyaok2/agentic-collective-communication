def evolved_p5200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # y = AllReduce_SUM(x) in fp16
    # Direct all_reduce on fp16 to avoid fp32 conversions
    return xm.all_reduce(xm.REDUCE_SUM, x)