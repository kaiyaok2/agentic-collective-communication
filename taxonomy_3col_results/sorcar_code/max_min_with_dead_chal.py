
def evolved_p5201(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    mn = xm.all_reduce(xm.REDUCE_MIN, y)
    return mx + mn
