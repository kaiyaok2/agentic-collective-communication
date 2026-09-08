
def evolved_p6901(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR(MAX) full: all-reduce with MAX on the full (32, 2048) tensor
    # Instead of 32 separate all_reduce calls (one per row),
    # do a single all_reduce on the entire tensor
    return xm.all_reduce(xm.REDUCE_MAX, x)
