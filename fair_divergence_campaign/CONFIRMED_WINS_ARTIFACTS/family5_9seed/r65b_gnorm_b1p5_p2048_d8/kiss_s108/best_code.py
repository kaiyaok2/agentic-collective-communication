
def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 1.5
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s / (1.0 + BETA * s.abs().mean())
    
    # Second all_reduce - after this result is identical on all ranks
    result = xm.all_reduce(xm.REDUCE_SUM, s) / W
    
    return result
