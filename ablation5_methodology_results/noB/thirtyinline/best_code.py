
def thirtyinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 30 identical ones
    t = xm.all_reduce(xm.REDUCE_SUM, x)
    # Multiply by 30 to get the same result as accumulating 30 identical all_reduces
    return 30.0 * t
