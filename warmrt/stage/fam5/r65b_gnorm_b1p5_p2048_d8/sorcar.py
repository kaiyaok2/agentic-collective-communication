
def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Fold world_size division into normalization
    g = (1.0 + 1.5 * s.abs().mean()) * world_size
    return xm.all_reduce(xm.REDUCE_SUM, s / g)
