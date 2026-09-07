
def evolved_p5001(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: 2*AR(x) + 3*AR(x) + 4*AR(x) + 6*AR(x) = 15*AR(x)
    # Optimization: single all-reduce instead of four
    ar_x = xm.all_reduce(xm.REDUCE_SUM, x)
    return ar_x * 15
