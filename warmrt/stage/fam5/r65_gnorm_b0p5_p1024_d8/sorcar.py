
def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s / (world_size * (1.0 + 0.5 * s.abs().mean()))
    return xm.all_reduce(xm.REDUCE_SUM, s)
