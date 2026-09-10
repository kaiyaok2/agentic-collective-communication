def seven_scaled_input_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Sum of 1+2+3+4+5+6+7 = 28
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 28.0 * reduced