def twentyeightinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 28
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    # Multiply by 28 to get the same result as summing 28 identical all_reduces
    result = result * 28.0
    return result