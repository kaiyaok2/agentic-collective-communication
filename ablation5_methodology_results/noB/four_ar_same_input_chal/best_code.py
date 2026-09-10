def evolved_p5001(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x) * 2
    b = xm.all_reduce(xm.REDUCE_SUM, x) * 3
    c = xm.all_reduce(xm.REDUCE_SUM, x) * 4
    d = xm.all_reduce(xm.REDUCE_SUM, x) * 6
    return a + b + c + d
