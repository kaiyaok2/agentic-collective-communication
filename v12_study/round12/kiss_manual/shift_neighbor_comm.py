def evolved_p114(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    pairs = [(r, (r - 1) % world_size) for r in range(world_size)]
    return xm.collective_permute(x, pairs=pairs)
