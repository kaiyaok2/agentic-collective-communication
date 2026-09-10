def evolved_p5502(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    if world_size >= 2:
        combined = x + y + z
        return xm.all_reduce(xm.REDUCE_SUM, combined)
    else:
        ax = xm.all_reduce(xm.REDUCE_SUM, x)
        return ax