
def fiftyinline_fn(x, N, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    scaled = 50.0 * x
    result = xm.all_reduce(xm.REDUCE_SUM, scaled)
    return result
