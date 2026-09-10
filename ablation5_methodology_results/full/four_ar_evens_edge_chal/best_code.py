def evolved_p4700(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Factor out 2: 2*(x1 + 2*x2 + 3*x3 + 4*x4)
    stacked = torch.stack([x1, 2*x2, 3*x3, 4*x4], dim=0)
    local_sum = 2 * torch.sum(stacked, dim=0)
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)