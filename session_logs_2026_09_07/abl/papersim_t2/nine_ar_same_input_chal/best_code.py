
def evolved_p6400(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: sum from k=1 to 9 of k*AR(x) = (1+2+...+9)*AR(x) = 45*AR(x)
    # Optimize: do 1 all-reduce instead of 9
    ar_x = xm.all_reduce(xm.REDUCE_SUM, x)
    return 45 * ar_x
