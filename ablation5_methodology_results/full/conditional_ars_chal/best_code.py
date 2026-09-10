def evolved_p5502(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    if world_size >= 2:
        # Add locally first, then all_reduce the sum
        local_sum = x + y + z
        return xm.all_reduce(xm.REDUCE_SUM, local_sum)
    else:
        # For single rank, just reduce x
        return xm.all_reduce(xm.REDUCE_SUM, x)
