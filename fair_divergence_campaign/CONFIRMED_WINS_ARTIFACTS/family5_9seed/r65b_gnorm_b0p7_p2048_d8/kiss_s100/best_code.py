
def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.7
    inv_W = 1.0 / world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Pre-compute inverse to use multiplication instead of division
    scale = inv_W / (1.0 + BETA * s.abs().mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, s * scale)
    
    return acc
