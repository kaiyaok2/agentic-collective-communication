def evolved_p3700(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    stacked = torch.stack([3*x, 5*y, 7*z], dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    return reduced.sum(dim=0)