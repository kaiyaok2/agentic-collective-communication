
def evolved_p5900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: AR(x) - compute all-reduce sum of x
    # Return (N,) identical on every rank
    return xm.all_reduce(xm.REDUCE_SUM, x)
