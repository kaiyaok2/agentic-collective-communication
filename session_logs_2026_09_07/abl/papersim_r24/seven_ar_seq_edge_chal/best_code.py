
def evolved_p4300(x1, x2, x3, x4, x5, x6, x7, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = AR(x1) + 2*AR(x2) + 3*AR(x3) + 4*AR(x4) + 5*AR(x5) + 6*AR(x6) + 7*AR(x7)
    # Optimization: AR is linear, so AR(x1) + 2*AR(x2) + ... = AR(x1 + 2*x2 + ...)
    
    # Compute weighted sum locally first
    local_sum = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 6*x6 + 7*x7
    
    # Single all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    
    return s
