
def evolved_p5602(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = AR(x1) + AR(x2) + AR(x3) + AR(x4) + AR(x5)
    # By linearity: AR(x1) + AR(x2) + AR(x3) + AR(x4) + AR(x5) = AR(x1+x2+x3+x4+x5)
    # Optimization: Add locally first, then do single all-reduce
    local_sum = x1 + x2 + x3 + x4 + x5
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)
