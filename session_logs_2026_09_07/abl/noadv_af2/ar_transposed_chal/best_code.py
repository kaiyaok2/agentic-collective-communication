def evolved_p5401(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Direct all-reduce without transposes
    return xm.all_reduce(xm.REDUCE_SUM, x)