def evolved_p3801(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine scaled tensors before reduction
    # 5 * all_reduce(z) + 3 * all_reduce(y) + 2 * all_reduce(x)
    # = all_reduce(5*z + 3*y + 2*x)
    combined = 5 * z + 3 * y + 2 * x
    result = xm.all_reduce(xm.REDUCE_SUM, combined)
    return result