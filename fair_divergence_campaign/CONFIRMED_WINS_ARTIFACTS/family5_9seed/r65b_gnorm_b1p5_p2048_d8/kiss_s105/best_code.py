
def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Try computing normalization factor differently
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    scale = 1.0 / (world_size * (1.0 + 1.5 * s.abs().mean()))
    return xm.all_reduce(xm.REDUCE_SUM, s) * scale
