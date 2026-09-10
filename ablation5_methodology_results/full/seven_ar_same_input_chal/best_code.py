
def evolved_p6300(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Optimized: single all_reduce instead of 7
    # 1 + 5 + 10 + 15 + 20 + 25 + 24 = 100
    return 100 * xm.all_reduce(xm.REDUCE_SUM, x)
