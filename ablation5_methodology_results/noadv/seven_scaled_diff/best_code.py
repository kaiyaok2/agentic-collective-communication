
def seven_scaled_diff_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All 7 all_reduce operations are on the same input x
    # Sum of scales: 1.0 + 2.0 - 1.0 + 3.0 + 4.0 - 3.0 + 5.0 = 11.0
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 11.0 * reduced
