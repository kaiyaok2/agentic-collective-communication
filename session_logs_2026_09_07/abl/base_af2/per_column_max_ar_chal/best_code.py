
def evolved_p7100(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce MAX on entire tensor at once
    # Element-wise max across all ranks: output[i,j] = max(x[i,j] over all ranks)
    # Single collective call instead of 16 per-column calls
    return xm.all_reduce(xm.REDUCE_MAX, x)
