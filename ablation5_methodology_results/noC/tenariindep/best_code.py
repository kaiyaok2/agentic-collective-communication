def tenariindep_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 10
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    # Multiply by 10 to match original behavior
    result = 10.0 * reduced
    return result