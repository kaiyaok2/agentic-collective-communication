def evolved_p117(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Need cumulative sum over ranks. all_gather then slice.
    g = xm.all_gather(x, dim=0).reshape(world_size, N)
    return g[:rank + 1].sum(dim=0)
