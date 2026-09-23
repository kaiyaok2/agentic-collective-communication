
def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Combine scale factors
    combined_scale = 1.0 / (world_size * (1.0 + BETA * s.abs().mean()))
    return xm.all_reduce(xm.REDUCE_SUM, s * combined_scale)
