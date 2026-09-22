
def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Try to compute the result of 4 all_reduces more efficiently
    # 4 all_reduces = sum across all ranks, repeated 4 times
    # This scales the sum by world_size^3
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    result = result * (world_size ** 3)
    return result
