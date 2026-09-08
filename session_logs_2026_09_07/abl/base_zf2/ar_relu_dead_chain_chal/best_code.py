
def evolved_p6102(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce x with SUM across all ranks
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    # Apply ReLU by multiplying with mask (ax > 0)
    return ax * (ax > 0).to(ax.dtype)
