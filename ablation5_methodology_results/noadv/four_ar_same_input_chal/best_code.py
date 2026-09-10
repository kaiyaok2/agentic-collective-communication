
def evolved_p5001(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return reduced * 15
