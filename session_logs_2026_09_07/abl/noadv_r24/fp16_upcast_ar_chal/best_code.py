
def evolved_p5200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Direct fp16 all_reduce - avoid dtype conversions
    return xm.all_reduce(xm.REDUCE_SUM, x)
