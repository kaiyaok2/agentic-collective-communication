def csescaleddiff_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine 5 all_reduce operations into 1
    # Original: 1.0*all_reduce + 2.0*all_reduce - 1.0*all_reduce + 3.0*all_reduce + 4.0*all_reduce
    # = (1.0 + 2.0 - 1.0 + 3.0 + 4.0) * all_reduce = 9.0 * all_reduce
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 9.0 * reduced