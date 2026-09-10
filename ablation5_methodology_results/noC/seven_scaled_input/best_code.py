
def seven_scaled_input_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute all_reduce once
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    # Sum of scales: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
    result = 28.0 * reduced
    return result
