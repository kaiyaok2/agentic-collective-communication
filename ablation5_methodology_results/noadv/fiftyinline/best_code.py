
def fiftyinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All 50 all_reduce operations on the same input x produce identical results
    # So: sum of 50 identical all_reduces = 50 * all_reduce(x)
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return 50.0 * result
