
def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    scale = 1.0 / ((1.0 + 0.5 * s.abs().mean()) * world_size)
    return xm.all_reduce(xm.REDUCE_SUM, s * scale)
