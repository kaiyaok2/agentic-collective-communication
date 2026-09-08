
def evolved_p5303(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = 2*AR(x) - AR(x) 
    # Simplifies to: y = AR(x)
    # Just call all_reduce once instead of twice
    return xm.all_reduce(xm.REDUCE_SUM, x)
