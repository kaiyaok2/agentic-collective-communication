
def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 3.0
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # After all_reduce, s is same on all ranks, so we can just return normalized value
    return s / (1.0 + BETA * s.abs().mean())
