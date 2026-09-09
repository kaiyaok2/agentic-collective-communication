def evolved_p5201(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute (max_r x_r) + (min_r y_r)
    # max_r x_r: element-wise maximum of x across all ranks
    # min_r y_r: element-wise minimum of y across all ranks
    # Result: (N,) tensor, identical on every rank
    
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    mn = xm.all_reduce(xm.REDUCE_MIN, y)
    return mx + mn