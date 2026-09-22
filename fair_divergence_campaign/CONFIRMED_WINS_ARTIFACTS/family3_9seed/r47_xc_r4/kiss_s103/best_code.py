
def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # 4 all_reduce operations compound: each one multiplies the sum by world_size
    # Result is world_size^3 * sum(x) across all ranks
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    scale = float(world_size ** 3)
    result = result * scale
    return result
