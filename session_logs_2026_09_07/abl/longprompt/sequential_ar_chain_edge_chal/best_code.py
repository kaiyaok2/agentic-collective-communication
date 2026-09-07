def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    scaled_x = 2 * world_size * x
    stacked = torch.stack([scaled_x, y], dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    return reduced.sum(dim=0)