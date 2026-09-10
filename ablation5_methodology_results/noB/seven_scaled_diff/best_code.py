
def seven_scaled_diff_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All 7 all_reduce operations are on the same tensor x
    # Sum of coefficients: 1 + 2 - 1 + 3 + 4 - 3 + 5 = 11
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 11.0 * reduced
