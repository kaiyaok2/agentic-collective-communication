
def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.5
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # After first all_reduce, all ranks have the same s
    # Second all_reduce will just multiply by world_size
    # So we can skip it and just return s/g
    g = 1.0 + BETA * s.abs().mean()
    return s / g
