
def evolved_p5103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute elementwise max of x across all ranks
    # Single all_reduce MAX is sufficient - max is idempotent so
    # max(max(x)) = max(x), making additional reductions redundant
    return xm.all_reduce(xm.REDUCE_MAX, x)
