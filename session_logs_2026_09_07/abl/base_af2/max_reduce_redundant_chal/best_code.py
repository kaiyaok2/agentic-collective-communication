
def evolved_p5000(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute elementwise max across all ranks: y[i] = max_r(x_r[i])
    # all_reduce with REDUCE_MAX does exactly this
    return xm.all_reduce(xm.REDUCE_MAX, x)
