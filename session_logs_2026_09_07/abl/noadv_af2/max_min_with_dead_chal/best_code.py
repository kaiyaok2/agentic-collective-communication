
def evolved_p5201(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute element-wise max of x across all ranks
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    # Compute element-wise min of y across all ranks
    mn = xm.all_reduce(xm.REDUCE_MIN, y)
    # Return their sum (identical on all ranks)
    return mx + mn
