
def r65b_gnorm_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.3
    # Try combining the operations more efficiently
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Compute normalization factor from the sum
    norm_factor = (1.0 + BETA * s.abs().mean()) * W
    # Normalize and do second all_reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, s / norm_factor)
    return acc
