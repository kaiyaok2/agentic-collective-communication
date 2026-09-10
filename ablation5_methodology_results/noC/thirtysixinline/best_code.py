
def thirtysixinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 36.0 * reduced
