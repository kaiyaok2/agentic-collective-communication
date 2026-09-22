
def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # 4 consecutive all_reduces = world_size^3 * sum(x)
    # More efficient: 1 all_reduce + local multiplication
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    scale = world_size ** 3
    result = s * scale
    return result
