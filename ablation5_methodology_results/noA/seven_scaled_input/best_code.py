def seven_scaled_input_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute sum of scales: 1+2+3+4+5+6+7 = 28
    # (1*all_reduce(x) + 2*all_reduce(x) + ... + 7*all_reduce(x))
    # = (1+2+3+4+5+6+7) * all_reduce(x) = 28 * all_reduce(x)
    return 28.0 * xm.all_reduce(xm.REDUCE_SUM, x)