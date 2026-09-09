
def evolved_p6502(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # y = 3*AR(x1) + 5*AR(x3) + 2*AR(x5)
    # Using linearity of all_reduce: AR(a*x + b*y) = a*AR(x) + b*AR(y)
    # So: 3*AR(x1) + 5*AR(x3) + 2*AR(x5) = AR(3*x1 + 5*x3 + 2*x5)
    
    # Combine locally first
    local_sum = 3 * x1 + 5 * x3 + 2 * x5
    
    # Single all_reduce
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)
