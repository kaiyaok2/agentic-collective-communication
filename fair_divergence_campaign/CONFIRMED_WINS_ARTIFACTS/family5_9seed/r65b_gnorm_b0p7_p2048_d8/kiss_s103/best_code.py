
def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.7
    W_inv = 1.0 / world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Final normalization
    g = 1.0 + BETA * s.abs().mean()
    s = xm.all_reduce(xm.REDUCE_SUM, s / g) * W_inv
    
    return s
