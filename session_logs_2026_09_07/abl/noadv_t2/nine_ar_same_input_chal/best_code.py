
def evolved_p6400(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: sum from k=1 to 9 of k*AR(x)
    # = (1+2+3+4+5+6+7+8+9) * AR(x)
    # = 45 * AR(x)
    # Single all-reduce instead of 9
    return 45 * xm.all_reduce(xm.REDUCE_SUM, x)
