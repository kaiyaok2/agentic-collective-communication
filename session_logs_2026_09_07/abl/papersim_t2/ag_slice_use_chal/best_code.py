
def evolved_p4802(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = 2 * x_r (scale own tensor locally)
    # No collective needed - purely local computation
    return 2 * x
