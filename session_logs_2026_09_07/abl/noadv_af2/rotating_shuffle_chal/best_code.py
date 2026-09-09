def evolved_p129(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    g = xm.all_gather(x, dim=0).reshape(world_size, N)
    plus1 = g[(rank + 1) % world_size]
    plus2 = g[(rank + 2) % world_size]
    return torch.cat([plus1, plus2], dim=0)
