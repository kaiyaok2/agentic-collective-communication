
def evolved_p4700(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = 2*AR(x1) + 4*AR(x2) + 6*AR(x3) + 8*AR(x4)
    # Exploit linearity: AR(2*x1 + 4*x2 + 6*x3 + 8*x4) = 2*AR(x1) + 4*AR(x2) + 6*AR(x3) + 8*AR(x4)
    
    # Compute weighted sum locally
    weighted_sum = 2 * x1 + 4 * x2 + 6 * x3 + 8 * x4
    
    # Single all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, weighted_sum)
    
    return s
