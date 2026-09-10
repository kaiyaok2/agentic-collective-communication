def evolved_p5502(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    if world_size <= 1:
        return x
    stacked = torch.stack([x, y, z], dim=0)
    local_sum = stacked.sum(dim=0)
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)