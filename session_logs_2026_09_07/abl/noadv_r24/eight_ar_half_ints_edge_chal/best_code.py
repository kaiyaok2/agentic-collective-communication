
def evolved_p4401(x1, x2, x3, x4, x5, x6, x7, x8, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = 0.5*AR(x1) + 1.5*AR(x2) + 2.5*AR(x3) + ... + 7.5*AR(x8)
    # Since AR is linear: AR(a+b) = AR(a) + AR(b) and AR(c*a) = c*AR(a)
    # We can rewrite as: s = AR(0.5*x1 + 1.5*x2 + ... + 7.5*x8)
    
    # Pre-scale and sum locally
    local_sum = 0.5 * x1 + 1.5 * x2 + 2.5 * x3 + 3.5 * x4 + 4.5 * x5 + 5.5 * x6 + 6.5 * x7 + 7.5 * x8
    
    # Single all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    
    return s
