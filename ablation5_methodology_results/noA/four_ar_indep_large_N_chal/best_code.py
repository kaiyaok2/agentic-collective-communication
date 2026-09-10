def evolved_p6403(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack and sum in one operation, then all_reduce
    stacked = torch.stack([x1, x2, x3, x4], dim=0)
    local_sum = torch.sum(stacked, dim=0)
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result