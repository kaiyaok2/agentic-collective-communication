def seven_scaled_input_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Mathematically equivalent: sum of (i * all_reduce(x)) for i in 1..7
    # = (1+2+3+4+5+6+7) * all_reduce(x) = 28 * all_reduce(x)
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    result = 28.0 * reduced
    return result