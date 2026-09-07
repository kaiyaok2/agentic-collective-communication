def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)     # bandwidth = world_size * N
    return ax.narrow(0, rank * N, N)         # only need N bytes but transferred W*N
