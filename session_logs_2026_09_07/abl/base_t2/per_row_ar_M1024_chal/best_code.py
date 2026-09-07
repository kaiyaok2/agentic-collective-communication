
def evolved_p7200(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: All-reduce SUM on the full tensor x (1024, 64)
    # Instead of 1024 separate all-reduce calls (one per row),
    # perform a single all-reduce on the entire tensor
    return xm.all_reduce(xm.REDUCE_SUM, x)
