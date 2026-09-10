def csescaleddiff_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute the sum once and scale by 9.0
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    result = 9.0 * result
    return result