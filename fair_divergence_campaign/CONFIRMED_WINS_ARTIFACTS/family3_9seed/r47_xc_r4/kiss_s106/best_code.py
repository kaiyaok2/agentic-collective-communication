
def r47_xc_r4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # After 4 all_reduce operations, the result is world_size^3 times the original sum
    # Try: 1 all_reduce + scalar multiplication
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    scale = float(world_size ** 3)
    result = result * scale
    return result
