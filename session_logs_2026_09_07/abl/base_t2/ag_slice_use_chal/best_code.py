
def evolved_p4802(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = 2 * x_r (elementwise scale of own tensor per rank)
    # No data from other ranks needed - just scale locally
    return 2 * x
