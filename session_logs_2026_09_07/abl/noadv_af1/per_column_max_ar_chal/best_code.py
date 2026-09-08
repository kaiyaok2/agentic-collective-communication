
def evolved_p7100(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR(MAX) full: All-reduce MAX on the entire (512, 16) tensor
    # Single collective call instead of 16 per-column calls
    return xm.all_reduce(xm.REDUCE_MAX, x)
