def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    norm_factor = (1.0 + 0.6 * s.abs().mean()) * world_size
    return xm.all_reduce(xm.REDUCE_SUM, s / norm_factor)
