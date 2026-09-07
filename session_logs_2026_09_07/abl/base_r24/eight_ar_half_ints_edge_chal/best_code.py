
def evolved_p4401(x1, x2, x3, x4, x5, x6, x7, x8, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = 0.5*AR(x1) + 1.5*AR(x2) + 2.5*AR(x3) + 3.5*AR(x4) + 4.5*AR(x5) + 5.5*AR(x6) + 6.5*AR(x7) + 7.5*AR(x8)
    # Since all-reduce is linear: AR(a+b) = AR(a) + AR(b) and AR(c*a) = c*AR(a)
    # Therefore: s = AR(0.5*x1 + 1.5*x2 + 2.5*x3 + 3.5*x4 + 4.5*x5 + 5.5*x6 + 6.5*x7 + 7.5*x8)
    
    weighted_sum = 0.5 * x1 + 1.5 * x2 + 2.5 * x3 + 3.5 * x4 + 4.5 * x5 + 5.5 * x6 + 6.5 * x7 + 7.5 * x8
    s = xm.all_reduce(xm.REDUCE_SUM, weighted_sum)
    return s
