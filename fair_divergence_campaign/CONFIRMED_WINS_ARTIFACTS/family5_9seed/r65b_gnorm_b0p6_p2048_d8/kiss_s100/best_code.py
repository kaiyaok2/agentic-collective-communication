
def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.6
    W = world_size
    
    # First all_reduce: compute global sum
    global_sum = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Normalize by gradient norm factor
    g = 1.0 + BETA * global_sum.abs().mean()
    normalized = global_sum / g
    
    # Since all ranks have the same value after all_reduce,
    # second all_reduce would just sum W copies of the same tensor
    # So we can skip it: W * normalized / W = normalized
    result = normalized
    
    return result
