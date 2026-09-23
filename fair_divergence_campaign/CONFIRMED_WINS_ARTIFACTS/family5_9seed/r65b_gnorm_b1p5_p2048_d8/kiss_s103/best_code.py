
def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 1.5
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Try different ordering: multiply by scalar first
    norm_factor = 1.0 / (1.0 + BETA * s.abs().mean()) / world_size
    s = xm.all_reduce(xm.REDUCE_SUM, s * norm_factor)
    return s
