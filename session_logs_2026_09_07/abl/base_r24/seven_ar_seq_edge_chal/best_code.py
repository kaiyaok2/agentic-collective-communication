
def evolved_p4300(x1, x2, x3, x4, x5, x6, x7, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Exploit linearity of all-reduce: AR(a+b) = AR(a) + AR(b)
    # So AR(x1) + 2*AR(x2) + ... + 7*AR(x7) = AR(x1 + 2*x2 + ... + 7*x7)
    # Compute weighted sum locally, then do a single all-reduce
    local_weighted_sum = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 6*x6 + 7*x7
    s = xm.all_reduce(xm.REDUCE_SUM, local_weighted_sum)
    return s
