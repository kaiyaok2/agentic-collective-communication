def evolved_p6402(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = 2*AR(x) + 3*AR(x) - AR(x) - 4*AR(x)
    # Coefficients: 2 + 3 - 1 - 4 = 0
    # So y = 0 * AR(x)
    ar = xm.all_reduce(xm.REDUCE_SUM, x)
    return 0 * ar