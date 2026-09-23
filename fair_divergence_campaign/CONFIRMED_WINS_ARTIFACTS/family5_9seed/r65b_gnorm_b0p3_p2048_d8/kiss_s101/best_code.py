
def r65b_gnorm_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # After all_reduce, all ranks have the same s, so just return normalized value
    return s / (1.0 + 0.3 * s.abs().mean())
