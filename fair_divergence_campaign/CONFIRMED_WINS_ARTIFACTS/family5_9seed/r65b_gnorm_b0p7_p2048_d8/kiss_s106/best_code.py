
def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + 0.7 * s.abs().mean()
    return s / g
