
def evolved_p5900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute AR(x): all-reduce sum of x across all ranks
    # Output: (N,) identical on every rank
    # The baseline redundantly computed two identical all-reduces (ar1, ar2)
    # and used torch.where with an all-True mask - all unnecessary.
    # Simply return the single all-reduce result.
    return xm.all_reduce(xm.REDUCE_SUM, x)
