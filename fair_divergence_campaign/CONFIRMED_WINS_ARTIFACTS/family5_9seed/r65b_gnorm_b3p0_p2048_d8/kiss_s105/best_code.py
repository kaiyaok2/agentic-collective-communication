
def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    return xm.all_reduce(xm.REDUCE_SUM, s / (world_size * (1.0 + 3.0 * s.abs().mean())))
