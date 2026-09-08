
def evolved_p5701(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute 2 * AR(x)
    # x is (128, 512), result is (128, 512)
    ar_x = xm.all_reduce(xm.REDUCE_SUM, x)
    return 2 * ar_x
