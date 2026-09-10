
def seven_scaled_input_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Optimization: since all_reduce returns the same value on all ranks,
    # we can compute (1+2+3+4+5+6+7) * all_reduce(x) = 28 * all_reduce(x)
    # instead of doing 7 separate all_reduce operations
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    result = result * 28.0
    return result
