
def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # No iterations, just final normalization
    am = s.abs().mean()
    s = xm.all_reduce(xm.REDUCE_SUM, s / (1.0 + 1.5 * am))
    return s / world_size
