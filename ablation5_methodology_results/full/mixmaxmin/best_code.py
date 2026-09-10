def mixmaxmin_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute global max and min once
    global_max = xm.all_reduce(xm.REDUCE_MAX, x)
    global_min = xm.all_reduce(xm.REDUCE_MIN, x)
    
    # Compute weighted sum locally
    # Sum of weights: 0.1 + 0.2 + ... + 0.8 = 3.6
    # Sum of 0.5*weights: 0.05 + 0.1 + ... + 0.4 = 1.8
    result = 3.6 * global_max + 1.8 * global_min
    
    return result