
def evolved_p4802(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = 2 * x_r (scale own tensor by 2)
    # No collective needed - each rank only needs its own input scaled
    return 2 * x
