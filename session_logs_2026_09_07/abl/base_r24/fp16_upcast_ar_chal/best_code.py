
def evolved_p5200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # y = AR(x) as fp16
    # Perform all-reduce directly on fp16 without conversions
    return xm.all_reduce(xm.REDUCE_SUM, x)
