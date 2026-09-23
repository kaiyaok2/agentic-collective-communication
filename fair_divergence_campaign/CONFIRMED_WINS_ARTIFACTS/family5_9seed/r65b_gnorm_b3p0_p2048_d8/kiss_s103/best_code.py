
def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 3.0
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Combine the normalization and world_size division
    combined_scale = world_size * (1.0 + BETA * s.abs().mean())
    return xm.all_reduce(xm.REDUCE_SUM, s) / combined_scale
