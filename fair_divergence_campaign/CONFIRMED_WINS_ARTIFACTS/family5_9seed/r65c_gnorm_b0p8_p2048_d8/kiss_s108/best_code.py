
def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.8
    
    # After first all_reduce, s is the same on all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Since s is synchronized, normalize it directly
    g = 1.0 + BETA * s.abs().mean()
    result = s / g
    
    return result
