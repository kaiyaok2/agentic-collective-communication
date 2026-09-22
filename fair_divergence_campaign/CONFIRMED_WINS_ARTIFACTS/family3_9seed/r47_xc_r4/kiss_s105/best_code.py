
def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    result = x
    for _ in range(4):
        result = xm.all_reduce(xm.REDUCE_SUM, result)
    return result
