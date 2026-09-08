def evolved_p6003(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = AR(x) + AR(zeros) + AR(zeros)
    # Since AR(zeros) = zeros, this simplifies to: y = AR(x)
    return xm.all_reduce(xm.REDUCE_SUM, x)